RSpec.describe Decisive::Decision do
  subject(:decision) { described_class.new(type: :choice, value: "billing", confidence: 0.8, distribution: {}, latency_ms: 1.0) }

  describe "#confident?" do
    it "defaults to a 0.7 threshold" do
      expect(decision).to be_confident
      expect(decision.tap { _1.confidence = 0.69 }).not_to be_confident
    end

    it "accepts a custom threshold" do
      expect(decision.confident?(0.9)).to be(false)
      expect(decision.confident?(0.8)).to be(true)
    end

    it "treats a nil confidence (as from score) as zero" do
      decision.confidence = nil
      expect(decision.confident?(0.0)).to be(true)
      expect(decision).not_to be_confident
    end
  end

  describe "#to_s" do
    it "returns the value as a string" do
      expect(decision.to_s).to eq("billing")
      expect(described_class.new(value: true).to_s).to eq("true")
    end
  end
end
