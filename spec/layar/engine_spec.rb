RSpec.describe Layar::Engine do
  subject(:engine) { described_class.new(config) }

  let(:config)   { Layar::Config.new }
  let(:scores)   { { "billing" => 0.7, "bug" => 0.2, "account" => 0.1 } }
  let(:pipeline) { FakePipeline.new(scores) }

  before { allow(Informers).to receive(:pipeline).and_return(pipeline) }

  describe "#warm!" do
    it "loads the configured zero-shot model once and returns self" do
      config.model = "some/model"
      expect(engine.warm!).to equal(engine)
      engine.warm!
      expect(Informers).to have_received(:pipeline).with("zero-shot-classification", "some/model").once
    end
  end

  describe "#choice" do
    let(:options) { %w[billing bug account] }

    it "returns the most likely option with its confidence and distribution" do
      d = engine.choice("charged twice", options:)
      expect(d).to have_attributes(type: :choice, value: "billing", confidence: 0.7)
      expect(d.distribution).to eq(scores)
      expect(d.latency_ms).to be_a(Float)
    end

    it "orders the distribution by the options given, not by score" do
      d = engine.choice("x", options: %w[account bug billing])
      expect(d.distribution.keys).to eq(%w[account bug billing])
    end

    it "stringifies and de-duplicates options" do
      engine.choice("x", options: [:billing, "billing", :bug])
      expect(pipeline.calls.last[:labels]).to eq(%w[billing bug])
    end

    it "calls the pipeline single-label with the configured template" do
      engine.choice("x", options:)
      expect(pipeline.calls.last).to include(input: "x", multi_label: false,
                                             hypothesis_template: "This example is about {}.")
    end

    it "accepts a per-call template" do
      engine.choice("x", options:, template: "The topic is {}.")
      expect(pipeline.calls.last[:hypothesis_template]).to eq("The topic is {}.")
    end

    it "stringifies the input" do
      engine.choice(nil, options:)
      expect(pipeline.calls.last[:input]).to eq("")
    end

    it "applies the configured temperature" do
      config.temperature = 2.0
      expect(engine.choice("x", options:).confidence).to be < 0.7
    end

    it "lets a per-call temperature override the config" do
      config.temperature = 2.0
      expect(engine.choice("x", options:, temperature: 1.0).confidence).to eq(0.7)
    end

    context "with a Hash of value => description" do
      let(:options) { { "billing" => "payments", bug: "something broken", "account" => "logging in" } }
      let(:scores)  { { "payments" => 0.1, "something broken" => 0.8, "logging in" => 0.1 } }

      it "shows the model the descriptions" do
        engine.choice("x", options:)
        expect(pipeline.calls.last[:labels]).to eq(["payments", "something broken", "logging in"])
      end

      it "returns and keys the distribution by the (stringified) values" do
        d = engine.choice("x", options:)
        expect(d.value).to eq("bug")
        expect(d.distribution).to eq("billing" => 0.1, "bug" => 0.8, "account" => 0.1)
      end

      it "rejects two values sharing one description" do
        expect { engine.choice("x", options: { "a" => "same", "b" => "same" }) }
          .to raise_error(ArgumentError, /duplicate descriptions: \["same"\]/)
      end
    end

    it "requires at least two options" do
      expect { engine.choice("x", options: %w[billing billing]) }
        .to raise_error(ArgumentError, /at least 2 options/)
    end

    it "enforces max_options" do
      config.max_options = 2
      expect { engine.choice("x", options:) }.to raise_error(ArgumentError, /coarse-to-fine/)
    end
  end

  describe "#bool" do
    let(:statement) { "This message is spam." }

    it "returns true when entailment >= 0.5" do
      pipeline = FakePipeline.new(statement => 0.8)
      allow(Informers).to receive(:pipeline).and_return(pipeline)

      d = engine.bool("win a cruise", statement:)
      expect(d).to have_attributes(type: :bool, value: true, confidence: 0.8)
      expect(d.distribution[true]).to eq(0.8)
      expect(d.distribution[false]).to be_within(1e-9).of(0.2)
    end

    it "returns false with the probability of false as confidence" do
      allow(Informers).to receive(:pipeline).and_return(FakePipeline.new(statement => 0.1))
      d = engine.bool("hi mum", statement:)
      expect(d.value).to be(false)
      expect(d.confidence).to be_within(1e-9).of(0.9)
    end

    it "treats exactly 0.5 as true" do
      allow(Informers).to receive(:pipeline).and_return(FakePipeline.new(statement => 0.5))
      expect(engine.bool("x", statement:).value).to be(true)
    end

    it "queries the statement verbatim, multi-label" do
      engine.bool("x", statement:)
      expect(pipeline.calls.last).to include(labels: [statement], multi_label: true, hypothesis_template: "{}")
    end

    context "with several statements" do
      let(:statements) { ["The sender is offering a prize.", "The message asks you to click a link."] }

      it "is true if any statement holds, with the strongest as confidence" do
        allow(Informers).to receive(:pipeline).and_return(FakePipeline.new(statements[0] => 0.1, statements[1] => 0.9))
        d = engine.bool("click here", statement: statements)
        expect(d).to have_attributes(value: true, confidence: 0.9)
      end

      it "is false when none hold" do
        allow(Informers).to receive(:pipeline).and_return(FakePipeline.new(statements[0] => 0.1, statements[1] => 0.2))
        d = engine.bool("hi mum", statement: statements)
        expect(d.value).to be(false)
        expect(d.confidence).to be_within(1e-9).of(0.8)
      end

      it "scores them independently in one multi-label call" do
        engine.bool("x", statement: statements)
        expect(pipeline.calls.size).to eq(1)
        expect(pipeline.calls.last).to include(labels: statements, multi_label: true)
      end
    end

    it "requires a statement" do
      expect { engine.bool("x", statement: []) }.to raise_error(ArgumentError, /at least 1 statement/)
    end

    it "applies temperature" do
      allow(Informers).to receive(:pipeline).and_return(FakePipeline.new(statement => 0.9))
      expect(engine.bool("x", statement:, temperature: 3.0).confidence).to be < 0.9
    end
  end

  describe "#score" do
    let(:criterion) { "The customer is angry." }

    before { allow(Informers).to receive(:pipeline).and_return(FakePipeline.new(criterion => 0.834567)) }

    it "returns the rounded entailment probability with no confidence" do
      d = engine.score("fix it now", criterion:)
      expect(d).to have_attributes(type: :score, value: 0.8346, confidence: nil)
      expect(d.distribution).to eq(criterion => 0.834567)
    end

    it "ignores the configured temperature" do
      config.temperature = 3.0
      expect(engine.score("x", criterion:).value).to eq(0.8346)
    end
  end

  describe "caching" do
    let(:cache) { MemoryCache.new }

    before do
      config.cache = cache
      config.cache_ttl = 60
    end

    it "caches pipeline results with the configured TTL" do
      2.times { engine.choice("same", options: %w[billing bug]) }
      expect(pipeline.calls.size).to eq(1)
      expect(cache.fetches.map { _1[:expires_in] }).to all(eq(60))
      expect(cache.store.keys).to all(start_with("layar:"))
    end

    it "uses distinct keys for different inputs, labels and models" do
      engine.choice("a", options: %w[billing bug])
      engine.choice("b", options: %w[billing bug])
      engine.choice("a", options: %w[billing account])
      config.model = "other/model"
      engine.choice("a", options: %w[billing bug])
      expect(cache.store.size).to eq(4)
    end

    it "does not touch a cache when none is configured" do
      config.cache = nil
      2.times { engine.choice("same", options: %w[billing bug]) }
      expect(pipeline.calls.size).to eq(2)
    end
  end

  it "accepts string-keyed pipeline output" do
    string_keyed = ->(*, **) { { "labels" => %w[bug billing], "scores" => [0.6, 0.4] } }
    allow(Informers).to receive(:pipeline).and_return(string_keyed)
    expect(engine.choice("x", options: %w[billing bug]).value).to eq("bug")
  end
end
