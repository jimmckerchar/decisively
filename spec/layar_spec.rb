RSpec.describe Layar do
  it "has a version number" do
    expect(Layar::VERSION).to match(/\A\d+\.\d+\.\d+/)
  end

  describe ".config / .configure" do
    it "has sensible defaults" do
      expect(described_class.config).to have_attributes(
        model: "Xenova/bart-large-mnli", temperature: 1.0, cache: nil, cache_ttl: 3600, max_options: 20,
        backend: :nli, laya_model: nil, question: "What is this about?"
      )
    end

    it "accepts :nli or :laya (as symbol or string) and rejects other backends" do
      described_class.config.backend = "laya"
      expect(described_class.config.backend).to eq(:laya)
      expect { described_class.config.backend = :gpt }.to raise_error(ArgumentError, /unknown backend :gpt/)
    end

    it "yields the config for mutation" do
      described_class.configure { |c| c.max_options = 5 }
      expect(described_class.config.max_options).to eq(5)
    end
  end

  describe "delegation to the engine" do
    let(:engine) { instance_double(Layar::Engine) }

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
      expect(described_class.engine).to be_a(Layar::Engine).and equal(described_class.engine)
    end
  end

  describe ".calibrate!" do
    # Model is 95% sure of "a" every time, but "a" is only right 6/10 times.
    let(:examples) do
      Array.new(10) { |i| { input: "text #{i}", options: %w[a b], answer: i < 6 ? "a" : :b } }
    end

    before do
      allow(Layar::ZeroShot).to receive(:load).and_return(FakePipeline.new("a" => 0.95, "b" => 0.05))
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

  describe ".calibrate_bool!" do
    let(:statement) { "Is the customer angry?" }
    # Laya-like: angry messages score 0.1-0.56, calm ones ~0.
    let(:scores) do
      { "a1" => 0.10, "a2" => 0.25, "a3" => 0.56, "a4" => 0.03, "a5" => 0.10, "a6" => 0.56,
        "c1" => 0.001, "c2" => 0.002, "c3" => 0.004, "c4" => 0.001, "c5" => 0.003, "c6" => 0.002 }
    end
    let(:examples) { scores.keys.map { |k| { input: k, answer: k.start_with?("a") } } }

    before do
      described_class.config.backend = :laya
      described_class.config.laya_model = "/models/laya"
      laya = Object.new
      scores_by_input = scores
      laya.define_singleton_method(:predict) do |input, questions|
        questions.transform_values { { false => 1 - scores_by_input.fetch(input), true => scores_by_input.fetch(input) } }
      end
      allow(Layar::Laya).to receive(:load).and_return(laya)
    end

    it "fits a cut-off, stores it for the statement and reports before/after" do
      result = described_class.calibrate_bool!(examples, statement:)

      expect(described_class.config.bool_calibrations[statement]).to eq(result.slice(:scale, :shift))
      expect(result[:accuracy_before]).to be_within(1e-3).of(8 / 12.0)
      expect(result[:accuracy_after]).to eq(1.0)
      expect(result[:threshold]).to be_between(0.004, 0.03)
      expect(result[:ece_after]).to be < result[:ece_before]
    end

    it "fits on raw probabilities even when a calibration already exists" do
      described_class.config.bool_calibrations[statement] = { scale: 5.0, shift: -9.0 }
      expect(described_class.calibrate_bool!(examples, statement:)[:accuracy_before]).to be_within(1e-3).of(8 / 12.0)
    end

    it "makes Layar.bool use the fit" do
      described_class.calibrate_bool!(examples, statement:)
      expect(described_class.bool("a1", statement:).value).to be(true)
      expect(described_class.bool("c3", statement:).value).to be(false)
    end
  end

  describe "calibration with a replaced engine" do
    it "stores fits on the running engine's config, not the module default" do
      custom = Layar::Config.new
      described_class.engine = Layar::Engine.new(custom)
      allow(Layar::ZeroShot).to receive(:load).and_return(FakePipeline.new("a" => 0.95, "b" => 0.05))

      described_class.calibrate!(Array.new(10) { |i| { input: "t#{i}", options: %w[a b], answer: i < 6 ? "a" : "b" } })

      expect(custom.temperature).to be > 1.0
      expect(described_class.config.temperature).to eq(1.0)
    end
  end
end

