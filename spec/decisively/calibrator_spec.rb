RSpec.describe Decisively::Calibrator do
  describe ".apply" do
    let(:dist) { { "a" => 0.7, "b" => 0.2, "c" => 0.1 } }

    it "returns the distribution unchanged for t = 1.0 or nil" do
      expect(described_class.apply(dist, 1.0)).to equal(dist)
      expect(described_class.apply(dist, nil)).to equal(dist)
    end

    it "flattens the distribution when t > 1" do
      out = described_class.apply(dist, 2.0)
      expect(out["a"]).to be < 0.7
      expect(out["c"]).to be > 0.1
    end

    it "sharpens the distribution when t < 1" do
      expect(described_class.apply(dist, 0.5)["a"]).to be > 0.7
    end

    it "keeps probabilities normalised and order-preserving" do
      out = described_class.apply(dist, 3.0)
      expect(out.values.sum).to be_within(1e-9).of(1.0)
      expect(out.keys).to eq(dist.keys)
      expect(out.max_by { |_, p| p }.first).to eq("a")
    end

    it "handles zero probabilities without blowing up" do
      out = described_class.apply({ "a" => 1.0, "b" => 0.0 }, 2.0)
      expect(out.values).to all(be_finite)
      expect(out.values.sum).to be_within(1e-9).of(1.0)
    end

    it "works with non-string keys (as used by bool)" do
      out = described_class.apply({ true => 0.9, false => 0.1 }, 2.0)
      expect(out[true]).to be_between(0.5, 0.9)
    end
  end

  describe ".nll" do
    it "is lower when the gold label gets more probability" do
      good = [[{ "a" => 0.9, "b" => 0.1 }, "a"]]
      bad  = [[{ "a" => 0.9, "b" => 0.1 }, "b"]]
      expect(described_class.nll(good, 1.0)).to be < described_class.nll(bad, 1.0)
    end
  end

  describe ".fit" do
    # Always 95% confident but only right 60% of the time -> overconfident.
    let(:overconfident) do
      Array.new(10) { |i| [{ "a" => 0.95, "b" => 0.05 }, i < 6 ? "a" : "b"] }
    end

    it "picks a temperature > 1 for overconfident predictions" do
      expect(described_class.fit(overconfident)).to be > 1.0
    end

    it "picks a temperature < 1 for underconfident predictions" do
      underconfident = Array.new(10) { [{ "a" => 0.6, "b" => 0.4 }, "a"] }
      expect(described_class.fit(underconfident)).to be < 1.0
    end

    it "returns a value from the grid, rounded to 2dp" do
      t = described_class.fit(overconfident, grid: [1.0, 2.0, 3.0])
      expect([1.0, 2.0, 3.0]).to include(t)
    end

    it "reduces ECE on the data it was fitted to" do
      t = described_class.fit(overconfident)
      after = overconfident.map { |d, g| [described_class.apply(d, t), g] }
      expect(described_class.ece(after)).to be < described_class.ece(overconfident)
    end
  end

  describe ".ece" do
    it "is zero when confidence matches accuracy exactly" do
      examples = [[{ "a" => 1.0, "b" => 0.0 }, "a"]] * 4
      expect(described_class.ece(examples)).to eq(0.0)
    end

    it "equals |confidence - accuracy| for a single bucket" do
      examples = Array.new(10) { |i| [{ "a" => 0.95, "b" => 0.05 }, i < 6 ? "a" : "b"] }
      expect(described_class.ece(examples)).to be_within(1e-9).of(0.35)
    end

    it "puts confidence 1.0 in the top bucket" do
      expect { described_class.ece([[{ "a" => 1.0 }, "a"]], bins: 5) }.not_to raise_error
    end
  end
end
