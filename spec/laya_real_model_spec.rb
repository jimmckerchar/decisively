# End-to-end checks for the Laya backend against real ONNX exports of Laya.
#   LAYAR_REAL_MODEL=1 LAYAR_LAYA_MODELS=/path/to/exports bundle exec rspec spec/laya_real_model_spec.rb
# LAYAR_LAYA_MODELS holds one directory per checkpoint (multilingual/, english/), each with
# model.onnx, tokenizer.json, tokenizer_config.json and rl_agent_config.json.
# Run this file on its own on small machines: each checkpoint holds ~2 GB while loaded.
models_dir = ENV.fetch("LAYAR_LAYA_MODELS", nil)

RSpec.describe Layar::Laya, :real_model do
  before { skip "set LAYAR_LAYA_MODELS to a directory of Laya ONNX exports" unless models_dir }

  # spec/fixtures/laya/<checkpoint>.json was recorded from Python laya 0.3.20: the token ids its
  # build_sequence produced and the probabilities its PyTorch model returned, for inputs chosen to
  # exercise truncation, option trimming, JSON state, non-ASCII text and stray mask tokens.
  Dir[File.expand_path("fixtures/laya/*.json", __dir__)].sort.each do |fixture_path|
    checkpoint = File.basename(fixture_path, ".json")

    context "#{checkpoint} checkpoint" do
      before(:all) do
        dir = models_dir && File.join(models_dir, checkpoint)
        @laya = described_class.load(dir) if dir && File.directory?(dir)
      end

      after(:all) do
        @laya = nil
        GC.start
      end

      before { skip "no #{checkpoint} export in LAYAR_LAYA_MODELS" unless @laya }

      JSON.parse(File.read(fixture_path))["cases"].each do |c|
        it "matches Python Laya: #{c['name']}" do
          questions = c["questions"].transform_values { |q| q.transform_keys(&:to_sym) }
          feeds = []
          session = @laya.instance_variable_get(:@session)
          allow(session).to receive(:run).and_wrap_original { |m, names, feed| feeds << feed; m.call(names, feed) }

          out = @laya.predict(c["state"], questions)

          feed = feeds.last
          c["input_ids"].each_with_index do |(qid, ids), row|
            expect(feed["input_ids"][row].first(feed["attention_mask"][row].sum)).to eq(ids), "token ids for #{qid}"
            expect(feed["marker_pos"][row].first(c["markers"][qid].size)).to eq(c["markers"][qid]), "markers for #{qid}"
          end
          c["probabilities"].each do |qid, expected|
            expect(out[qid].values).to match(expected.map { |p| be_within(1e-5).of(p) }), "probabilities for #{qid}"
          end
        end
      end
    end
  end
end

RSpec.describe "Layar with the Laya backend (multilingual)", :real_model do
  before(:all) do
    dir = models_dir && File.join(models_dir, "multilingual")
    if dir && File.directory?(dir)
      config = Layar::Config.new.tap { |c| c.backend = :laya; c.laya_model = dir }
      @engine = Layar::Engine.new(config).warm!
    end
  end

  after(:all) do
    @engine = nil
    GC.start
  end

  before do
    skip "no multilingual export in LAYAR_LAYA_MODELS" unless @engine
    Layar.engine = @engine
  end

  it "routes a double charge to billing" do
    d = Layar.choice("I was charged twice for my subscription this month", options: %w[billing bug feature_request account])
    expect(d.value).to eq("billing")
  end

  spam = ["CONGRATS!!! You won a free cruise, click here",
          "URGENT: your account has been suspended. Verify your password at http://secure-login.co",
          "Make $5000 a week working from home, no experience needed!!!",
          "Final notice: claim your unclaimed parcel now by paying the $1.99 fee"]
  # The ordinary messages bart-large-mnli flags, because they share a topic with spam.
  ham = ["Hi, can you send me the invoice for last month?",
         "Could you click the link in my last email and confirm the meeting time?",
         "Please reset my password, I can't log in to my account.",
         "Thanks for the quick fix yesterday, everything works now.",
         "Can you tell me how to download my receipts?"]
  question = "Is this message spam, phishing or a scam?"

  spam.each do |text|
    it "flags spam: #{text[0, 40]}…" do
      expect(Layar.bool(text, statement: question).value).to be(true)
    end
  end

  ham.each do |text|
    it "does not flag: #{text[0, 40]}…" do
      expect(Layar.bool(text, statement: question).value).to be(false)
    end
  end
end
