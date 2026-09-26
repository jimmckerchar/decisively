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
end
