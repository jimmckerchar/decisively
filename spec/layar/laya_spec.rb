require "tmpdir"

RSpec.describe Layar::Laya do
  subject(:laya) do
    described_class.new(session:, tokenizer:, special_tokens:, config: { "max_len" => max_len, "head_max_len" => head_max_len,
                                                                            "temperature" => temperature,
                                                                            "temperature_by_options" => temperature_by_options })
  end

  # One id per whitespace-separated word, assigned on first sight; specials are fixed.
  let(:tokenizer) do
    Class.new do
      SPECIALS = { "<cls>" => 1, "<sep>" => 2, "<mask>" => 3, "<pad>" => 0 }.freeze

      def initialize = @vocab = {}
      def token_to_id(token) = SPECIALS[token]
      def id(word) = (@vocab[word] ||= 100 + @vocab.size)

      def encode(text, add_special_tokens:)
        raise "Laya encodes without special tokens" if add_special_tokens
        Struct.new(:ids).new(text.split.map { |w| id(w) })
      end
    end.new
  end

  # Records the batch it was given and returns fixed logits per question row.
  let(:session) do
    Class.new do
      attr_reader :feeds
      attr_accessor :logits

      def initialize = @feeds = []

      def run(output_names, feed)
        raise "unexpected outputs #{output_names}" unless output_names == %w[logits act_logits]
        @feeds << feed
        [@logits || feed["marker_mask"].map { |row| row.map { 0.0 } }, feed["qtype"].map { [5.0, -5.0] }]
      end
    end.new
  end

  let(:special_tokens) { { "cls_token" => "<cls>", "sep_token" => "<sep>", "mask_token" => "<mask>", "pad_token" => "<pad>" } }
  let(:max_len)                { 64 }
  let(:head_max_len)           { 32 }
  let(:temperature)            { [1.0, 1.0, 1.0] }
  let(:temperature_by_options) { {} }

  def w(word) = tokenizer.id(word)
  def feed = session.feeds.last

  describe "sequence layout" do
    it "is [CLS] <type> question: <instructions> [SEP] [MASK] opt ... [SEP] state [SEP]" do
      laya.predict("charged twice", q: { type: :choice, instructions: "Which team?", criteria: %w[billing bug] })

      expect(feed["input_ids"]).to eq([[1, w("choice"), w("question:"), w("Which"), w("team?"), 2,
                                        3, w("billing"), 3, w("bug"), 2,
                                        w("charged"), w("twice"), 2]])
      expect(feed["marker_pos"]).to eq([[6, 8]])
      expect(feed["marker_mask"]).to eq([[true, true]])
      expect(feed["attention_mask"]).to eq([[1] * 14])
      expect(feed["qtype"]).to eq([0])
    end

    it "renders choice descriptions as 'option: description'" do
      laya.predict("x", q: { type: :choice, instructions: "Q", criteria: { "billing" => "payments", "bug" => nil } })
      expect(feed["input_ids"][0][5..9]).to eq([3, w("billing:"), w("payments"), 3, w("bug")])
    end

    it "renders JSON-valued descriptions like Python's json.dumps" do
      laya.predict("x", q: { type: :choice, instructions: "Q", criteria: { "a" => { "desc" => "pay", "n" => [1, 2] }, "b" => nil } })
      expect(feed["input_ids"][0][5..10]).to eq([3, w("a:"), w('{"desc":'), w('"pay",'), w('"n":'), w("[1,")])
    end

    it "renders noul options with default descriptions, false first" do
      laya.predict("x", q: { type: :noul, instructions: "Spam?" })
      expect(feed["input_ids"][0][5..]).to start_with(
        3, w("false:"), w("no,"), w("the"), w("statement"), w("does"), w("not"), w("hold"),
        3, w("true:"), w("yes,"), w("the"), w("statement"), w("holds")
      )
      expect(feed["qtype"]).to eq([2])
    end

    it "accepts noul descriptions keyed by booleans" do
      laya.predict("x", q: { type: :noul, instructions: "Q", criteria: { true => "spam", false => "fine" } })
      expect(feed["input_ids"][0][5..10]).to eq([3, w("false:"), w("fine"), 3, w("true:"), w("spam")])
    end

    it "renders score levels as 'level i: description'" do
      laya.predict("x", q: { type: :score, instructions: "Q", criteria: %w[calm furious] })
      expect(feed["input_ids"][0][5..12]).to eq([3, w("level"), w("0:"), w("calm"), 3, w("level"), w("1:"), w("furious")])
      expect(feed["qtype"]).to eq([1])
    end

    it "strips mask tokens out of text so they can't pose as markers" do
      laya.predict("a <mask> b", q: { type: :choice, instructions: "<mask>?", criteria: ["x <mask>", "y"] })
      expect(feed["input_ids"][0].count(3)).to eq(2)
    end

    it "sends Hash and Array state as JSON" do
      laya.predict({ "subject" => "Refund", "n" => 2 }, q: { type: :noul, instructions: "Q" })
      expect(feed["input_ids"][0]).to include(w('{"subject":'), w('"Refund",'), w('"n":'), w("2}"))
    end
  end

  describe "truncation" do
    # The noul question and its two options take 21 tokens, leaving room for 18 of the 30 state words.
    let(:max_len) { 40 }
    let(:state)   { (1..30).map { |i| "s#{i}" } }

    it "keeps the start of a String state" do
      laya.predict(state.join(" "), q: { type: :noul, instructions: "Q" })
      ids = feed["input_ids"][0]
      expect(ids.size).to eq(40)
      expect(ids.last).to eq(2)
      expect(ids).to include(w("s1")).and(satisfy { |i| !i.include?(w("s30")) })
    end

    it "keeps the end of an Array (conversation) state" do
      laya.predict(state, q: { type: :noul, instructions: "Q" })
      expect(feed["input_ids"][0]).to include(w('"s30"]')).and(satisfy { |i| !i.include?(w('["s1",')) })
    end

    context "with many long options" do
      let(:max_len)      { 512 }
      let(:head_max_len) { 40 }

      it "trims each option to fit the head budget" do
        options = (1..6).map { |i| (["opt#{i}"] + %w[w] * 20).join(" ") }
        laya.predict("x", q: { type: :choice, instructions: "Q", criteria: options })
        per = (40 - 16) / 6
        markers = feed["marker_pos"][0]
        expect(markers.each_cons(2).map { |a, b| b - a }).to all(eq(per))
      end
    end

    it "raises when options can't fit" do
      big = described_class.new(session:, tokenizer:, special_tokens:, config: { "max_len" => 20, "head_max_len" => 64 })
      expect { big.predict("x", q: { type: :choice, instructions: "Q", criteria: (1..10).map { |i| "o#{i}" } }) }
        .to raise_error(ArgumentError, /exceed the 64-token head budget/)
    end
  end

  describe "batching" do
    it "sends every question in one padded model call" do
      laya.predict("x", a: { type: :choice, instructions: "Which one?", criteria: %w[p q r] },
                        b: { type: :noul, instructions: "Q" })

      expect(session.feeds.size).to eq(1)
      ids, mask = feed["input_ids"], feed["attention_mask"]
      expect(ids.map(&:size).uniq.size).to eq(1)
      short = mask.map(&:sum).min
      expect(ids[mask.map(&:sum).index(short)].drop(short)).to all(eq(0))
      expect(feed["marker_mask"]).to eq([[true, true, true], [true, true, false]])
      expect(feed["marker_pos"][1][2]).to eq(0)
      expect(feed["qtype"]).to eq([0, 2])
    end
  end

  describe "outputs" do
    before { session.logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]] }

    let(:questions) do
      { team: { type: :choice, instructions: "Q", criteria: { "billing" => "pay", "bug" => nil, "other" => nil } },
        spam: { type: :noul, instructions: "Q" } }
    end

    def softmax(xs) = xs.map { |x| Math.exp(x) / xs.sum { |y| Math.exp(y) } }

    it "returns option probabilities keyed like the criteria, ignoring padded logits" do
      out = laya.predict("x", questions)
      expect(out.keys).to eq(%i[team spam])
      expect(out[:team].keys).to eq(%w[billing bug other])
      expect(out[:team].values).to match(softmax([2.0, 0.0, -1.0]).map { |p| be_within(1e-12).of(p) })
      expect(out[:spam].keys).to eq([false, true])
      expect(out[:spam].values).to match(softmax([0.0, 1.0]).map { |p| be_within(1e-12).of(p) })
    end

    it "keys score probabilities by level" do
      session.logits = [[0.0, 0.0, 0.0]]
      out = laya.predict("x", q: { type: :score, instructions: "Q", criteria: %w[a b c] })
      expect(out[:q].keys).to eq([0, 1, 2])
    end

    context "with checkpoint temperatures" do
      let(:temperature)            { [2.0, 1.0, 4.0] }
      let(:temperature_by_options) { { "choice:3-5" => 0.5 } }

      it "prefers the per-option-count bucket, then the per-type value" do
        out = laya.predict("x", questions)
        expect(out[:team].values).to match(softmax([4.0, 0.0, -2.0]).map { |p| be_within(1e-12).of(p) })
        expect(out[:spam].values).to match(softmax([0.0, 0.25]).map { |p| be_within(1e-12).of(p) })
      end
    end

    context "with out-of-range temperatures" do
      let(:temperature_by_options) { { "choice:3-5" => 0.1, "noul:2" => "nonsense" } }

      it "clamps them to 0.5..5.0, or 1.0 when not a number" do
        out = laya.predict("x", questions)
        expect(out[:team].values).to match(softmax([4.0, 0.0, -2.0]).map { |p| be_within(1e-12).of(p) })
        expect(out[:spam].values).to match(softmax([0.0, 1.0]).map { |p| be_within(1e-12).of(p) })
      end
    end
  end

  it "returns {} without calling the model when there are no questions" do
    expect(laya.predict("x", {})).to eq({})
    expect(session.feeds).to be_empty
  end

  it "rejects unknown question types and empty criteria" do
    expect { laya.predict("x", q: { type: :rank, instructions: "Q" }) }.to raise_error(ArgumentError, /unknown type :rank/)
    expect { laya.predict("x", q: { type: :choice, instructions: "Q", criteria: [] }) }.to raise_error(ArgumentError, /needs criteria/)
    expect { laya.predict("x", q: { type: :score, instructions: "Q" }) }.to raise_error(ArgumentError, /list of levels/)
  end

  describe ".download" do
    let(:hub) { Informers::Utils::Hub }

    before do
      allow(hub).to receive(:get_model_file) { |repo, file, *| "/cache/#{repo}/#{file}" }
    end

    it "fetches every file of a named checkpoint from Layar's Hub repo, at the pinned revision" do
      expect(described_class.download("multilingual")).to eq("/cache/distinctinteractive/laya-onnx/multilingual")
      described_class::FILES.each do |file|
        expect(hub).to have_received(:get_model_file)
          .with("distinctinteractive/laya-onnx", "multilingual/#{file}", true,
                revision: described_class::HUB_REVISION, progress_callback: anything)
      end
    end

    it "pins a full commit hash" do
      expect(described_class::HUB_REVISION).to match(/\A\h{40}\z/)
    end

    it "accepts owner/repo/subfolder, following main" do
      expect(described_class.download("acme/laya-exports/v2/english")).to eq("/cache/acme/laya-exports/v2/english")
      expect(hub).to have_received(:get_model_file)
        .with("acme/laya-exports", "v2/english/model.onnx", true, revision: "main", progress_callback: anything)
    end

    it "explains what it accepts otherwise" do
      ["spanish", "/missing/dir", "acme/laya"].each do |bad|
        expect { described_class.download(bad) }.to raise_error(ArgumentError, /not a local directory, a checkpoint/)
      end
    end
  end

  describe ".load" do
    it "uses a local directory as-is, without downloading" do
      allow(described_class).to receive(:download)
      allow(OnnxRuntime::InferenceSession).to receive(:new).and_return(session)
      allow(Tokenizers).to receive(:from_file).and_return(tokenizer)
      Dir.mktmpdir do |dir|
        File.write(File.join(dir, "tokenizer_config.json"), JSON.dump(special_tokens))
        File.write(File.join(dir, "rl_agent_config.json"), JSON.dump({ "encoder" => "enc", "model_name" => "m" }))

        expect(described_class.load(dir).identity).to eq("enc/m")
        expect(described_class).not_to have_received(:download)
        expect(OnnxRuntime::InferenceSession).to have_received(:new).with(File.join(dir, "model.onnx"))
      end
    end
  end
end
