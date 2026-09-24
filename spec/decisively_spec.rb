RSpec.describe Decisively do
  it "has a version number" do
    expect(Decisively::VERSION).to match(/\A\d+\.\d+\.\d+/)
  end

  describe ".config / .configure" do
    it "has sensible defaults" do
      expect(described_class.config).to have_attributes(
        model: "Xenova/mobilebert-uncased-mnli", temperature: 1.0, cache: nil, cache_ttl: 3600, max_options: 20
      )
    end

    it "yields the config for mutation" do
      described_class.configure { |c| c.max_options = 5 }
      expect(described_class.config.max_options).to eq(5)
    end
  end

  describe "delegation to the engine" do
    let(:engine) { instance_double(Decisively::Engine) }

    before { described_class.engine = engine }

    it "delegates choice, bool, score and warm!" do
      allow(engine).to receive_messages(choice: :c, bool: :b, score: :s, warm!: engine)

      expect(described_class.choice("t", options: %w[a b], temperature: 2.0)).to eq(:c)
      expect(engine).to have_received(:choice).with("t", options: %w[a b], temperature: 2.0)

      expect(described_class.bool("t", statement: "S.")).to eq(:b)
      expect(engine).to have_received(:bool).with("t", statement: "S.")

      expect(described_class.score("t", criterion: "C.")).to eq(:s)
      expect(engine).to have_received(:score).with("t", criterion: "C.")

      expect(described_class.warm!).to eq(engine)
    end
  end

  describe ".engine" do
    it "builds and memoises an engine from the config" do
      expect(described_class.engine).to be_a(Decisively::Engine).and equal(described_class.engine)
    end
  end

  describe ".calibrate!" do
    # Model is 95% sure of "a" every time, but "a" is only right 6/10 times.
    let(:examples) do
      Array.new(10) { |i| { input: "text #{i}", options: %w[a b], answer: i < 6 ? "a" : :b } }
    end

    before do
      allow(Informers).to receive(:pipeline).and_return(FakePipeline.new("a" => 0.95, "b" => 0.05))
    end

    it "fits a temperature, stores it in config and reports ECE before/after" do
      result = described_class.calibrate!(examples)

      expect(result[:temperature]).to be > 1.0
      expect(described_class.config.temperature).to eq(result[:temperature])
      expect(result[:ece_before]).to eq(0.35)
      expect(result[:ece_after]).to be < result[:ece_before]
    end

    it "measures raw probabilities even when a temperature is already set" do
      described_class.config.temperature = 4.0
      expect(described_class.calibrate!(examples)[:ece_before]).to eq(0.35)
    end
  end
end
