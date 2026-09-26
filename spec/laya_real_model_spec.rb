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

RSpec.describe "Layar.calibrate_bool! with the Laya backend (multilingual)", :real_model do
  before(:all) do
    dir = ENV["LAYAR_LAYA_MODELS"] && File.join(ENV["LAYAR_LAYA_MODELS"], "multilingual")
    @engine = Layar::Engine.new(Layar::Config.new.tap { |c| c.backend = :laya; c.laya_model = dir }).warm! if dir && File.directory?(dir)
  end

  after(:all) do
    @engine = nil
    GC.start
  end

  before do
    skip "no multilingual export in LAYAR_LAYA_MODELS" unless @engine
    Layar.engine = @engine
  end

  # Laya ranks angry above calm but scores anger low, so at 0.5 it misses most angry messages.
  # A cut-off fitted on one set of messages should carry over to messages it has not seen.
  train = { true  => ["This is the third time I've asked. Fix it now.",
                      "I've been waiting two weeks for a refund and nobody replies. Unacceptable.",
                      "WHY IS THE APP STILL BROKEN?? I pay for this!!",
                      "Great, another update that deletes my data. Thanks a lot.",
                      "If this isn't sorted by Friday I'm cancelling and telling everyone to avoid you.",
                      "Your support is useless. I want to speak to a manager."],
            false => ["Thanks so much, that fixed it!", "Hi, can you send me the invoice for last month?",
                      "The export button doesn't seem to work on Safari, is that a known issue?",
                      "Could we move our call to Thursday?", "I'm a bit confused about how the pricing tiers work.",
                      "Love the new dashboard, great job team!"] }
  held_out = { true  => ["I am extremely disappointed. I was promised a callback on Monday and heard nothing.",
                         "Oh wonderful, charged AGAIN for a plan I cancelled. Brilliant service.",
                         "Per my last three emails, the invoice is still wrong. Please escalate this immediately.",
                         "This is ridiculous. Your app logged me out mid-payment and took the money anyway.",
                         "Do not contact me again until you have an actual answer.",
                         "Absolutely fed up with the constant outages. We're looking at other providers."],
               false => ["Quick question: does the Pro plan include API access?",
                         "Hi! Just letting you know the typo on the pricing page, no rush.",
                         "I think I found a bug: the date picker shows the wrong month. Screenshot attached.",
                         "Thank you for the refund, received it today.",
                         "Can I add a second user to my account?",
                         "Not sure if this is expected, but the report took a while to load this morning."] }

  it "improves anger detection on held-out messages" do
    statement = "Is the customer angry?"
    options = { yes: "the customer is angry, frustrated, impatient or demanding",
                no: "the customer is calm, neutral, polite or happy" }
    correct = -> { held_out.sum { |answer, texts| texts.count { |t| Layar.bool(t, statement:, **options).value == answer } } }

    Layar.config.bool_calibrations.clear
    before_fit = correct.()
    Layar.calibrate_bool!(train.flat_map { |answer, texts| texts.map { |t| { input: t, answer: } } }, statement:, **options)
    after_fit = correct.()

    expect(after_fit).to be > before_fit
    expect(after_fit).to be >= 10
  ensure
    Layar.config.bool_calibrations.clear
  end
end

