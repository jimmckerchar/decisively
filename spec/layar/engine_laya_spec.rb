RSpec.describe Layar::Engine, "with the Laya backend" do
  subject(:engine) { described_class.new(config) }

  let(:config) do
    Layar::Config.new.tap do |c|
      c.backend    = :laya
      c.laya_model = "/models/laya-multilingual"
    end
  end
  let(:laya) { FakeLaya.new(answers) }
  let(:answers) { {} }

  before { allow(Layar::Laya).to receive(:load).and_return(laya) }

  describe "#warm!" do
    it "loads the Laya model from laya_model, not the NLI model" do
      allow(Layar::ZeroShot).to receive(:load)
      engine.warm!
      expect(Layar::Laya).to have_received(:load).with("/models/laya-multilingual").once
      expect(Layar::ZeroShot).not_to have_received(:load)
    end

    it "explains what to set when laya_model is missing" do
      config.laya_model = nil
      expect { engine.warm! }.to raise_error(Layar::Error, /c.laya_model/)
    end
  end

  describe "#choice" do
    let(:answers) { { "What is this about?" => { "billing" => 0.7, "bug" => 0.2, "account" => 0.1 } } }

    it "asks one choice question and returns its distribution" do
      d = engine.choice("charged twice", options: %w[billing bug account])

      expect(d).to have_attributes(type: :choice, value: "billing", confidence: 0.7)
      expect(d.distribution).to eq("billing" => 0.7, "bug" => 0.2, "account" => 0.1)
      expect(laya.calls).to eq([{ state: "charged twice",
                                  questions: { choice: { type: :choice, instructions: "What is this about?",
                                                         criteria: { "billing" => nil, "bug" => nil, "account" => nil } } } }])
    end

    it "sends Hash descriptions as Laya criteria descriptions" do
      engine.choice("x", options: { "billing" => "payments", bug: "something broken", "account" => "account" })
      expect(laya.calls.last[:questions][:choice][:criteria])
        .to eq("billing" => "payments", "bug" => "something broken", "account" => nil)
    end

    it "uses a per-call question, then the configured default" do
      engine.choice("x", options: %w[a b], question: "Which team should handle this?")
      config.question = "Which queue?"
      engine.choice("x", options: %w[a b])
      expect(laya.calls.map { _1[:questions][:choice][:instructions] }).to eq(["Which team should handle this?", "Which queue?"])
    end

    it "still applies Layar's temperature on top" do
      config.temperature = 2.0
      expect(engine.choice("x", options: %w[billing bug account]).confidence).to be < 0.7
    end

    it "caches by model directory, input and question" do
      config.cache = MemoryCache.new
      2.times { engine.choice("same", options: %w[billing bug]) }
      engine.choice("other", options: %w[billing bug])
      expect(laya.calls.size).to eq(2)
    end
  end

  describe "#bool" do
    let(:answers) do
      { "The sender is offering a prize." => { false => 0.9, true => 0.1 },
        "Is this message spam?"           => { false => 0.2, true => 0.8 } }
    end

    it "asks each statement as a yes/no question in one call and takes the strongest" do
      d = engine.bool("win a cruise", statement: ["The sender is offering a prize.", "Is this message spam?"])

      expect(d).to have_attributes(type: :bool, value: true, confidence: 0.8)
      expect(laya.calls.size).to eq(1)
      expect(laya.calls.last[:questions]).to eq(
        0 => { type: :noul, instructions: "The sender is offering a prize." },
        1 => { type: :noul, instructions: "Is this message spam?" }
      )
    end

    it "is false when no statement holds" do
      d = engine.bool("hi", statement: "The sender is offering a prize.")
      expect(d.value).to be(false)
      expect(d.confidence).to be_within(1e-9).of(0.9)
    end
  end

  describe "#score" do
    let(:answers) { { "The customer is angry." => { false => 0.25, true => 0.75 } } }

    it "returns the probability that the criterion holds" do
      d = engine.score("fix it now", criterion: "The customer is angry.")
      expect(d).to have_attributes(type: :score, value: 0.75, confidence: nil)
    end
  end

  describe "yes:/no: descriptions" do
    it "sends them to Laya as noul criteria for bool and score" do
      engine.bool("x", statement: "Is the customer angry?", yes: "angry or impatient", no: "calm or happy")
      engine.score("x", criterion: "Is the customer angry?", yes: "angry or impatient")

      expect(laya.calls.map { _1[:questions][0] }).to eq([
        { type: :noul, instructions: "Is the customer angry?", criteria: { true => "angry or impatient", false => "calm or happy" } },
        { type: :noul, instructions: "Is the customer angry?", criteria: { true => "angry or impatient" } },
      ])
    end

    it "leaves criteria out when none are given" do
      engine.bool("x", statement: "Is the customer angry?")
      expect(laya.calls.last[:questions][0]).not_to have_key(:criteria)
    end
  end

  describe "bool calibration" do
    let(:answers) { { "Is the customer angry?" => { false => 0.7, true => 0.3 } } }

    before { config.bool_calibrations["Is the customer angry?"] = { scale: 1.0, shift: 2.0 } }

    it "applies the fit stored for the statement" do
      d = engine.bool("x", statement: "Is the customer angry?")
      expected = Layar::Calibrator.apply_platt(0.3, scale: 1.0, shift: 2.0)
      expect(d.value).to be(true)
      expect(d.distribution[true]).to be_within(1e-9).of(expected)
    end

    it "is skipped with calibrate: false" do
      expect(engine.bool("x", statement: "Is the customer angry?", calibrate: false).distribution[true]).to eq(0.3)
    end

    it "only applies to the statement it was fitted for" do
      expect(engine.bool("x", statement: "Is the customer happy?").distribution[true]).to eq(0.5)
    end

    context "with a fingerprinted fit" do
      let(:fit) { { scale: 1.0, shift: 2.0, fitted_for: engine.calibration_fingerprint(yes: "angry", no: "calm") } }

      before { config.bool_calibrations["Is the customer angry?"] = fit }

      it "applies it when the model and descriptions match" do
        d = engine.bool("x", statement: "Is the customer angry?", yes: "angry", no: "calm")
        expect(d.value).to be(true)
      end

      it "warns once and skips it when the descriptions changed" do
        expect {
          2.times { expect(engine.bool("x", statement: "Is the customer angry?", yes: "furious", no: "calm").value).to be(false) }
        }.to output(/ignoring the calibration for "Is the customer angry\?".*Refit/).to_stderr
        expect { engine.bool("x", statement: "Is the customer angry?", yes: "furious", no: "calm") }.not_to output.to_stderr
      end

      it "skips it when the model changed" do
        fit # fingerprint taken with the original model
        laya.identity = "another-encoder/rl-agent/9"
        expect { expect(engine.bool("x", statement: "Is the customer angry?", yes: "angry", no: "calm").value).to be(false) }
          .to output(/different model/).to_stderr
      end
    end

    it "fingerprints the NLI model and Laya checkpoint differently" do
      laya_print = engine.calibration_fingerprint
      config.backend = :nli
      expect(described_class.new(config).calibration_fingerprint).not_to eq(laya_print)
    end
  end
end

