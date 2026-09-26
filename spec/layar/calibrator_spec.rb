RSpec.describe Layar::Calibrator do
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

  describe ".fit_platt / .apply_platt" do
    # Ranks perfectly but scores low: every "yes" is below 0.5, every "no" near 0.
    let(:low_but_ranked) do
      [0.10, 0.25, 0.56, 0.03, 0.10, 0.56].map { [_1, true] } + [0.001, 0.002, 0.004, 0.001, 0.003, 0.002].map { [_1, false] }
    end

    def accuracy(pairs) = pairs.count { |p, yes| (p >= 0.5) == yes }

    it "moves the cut-off so low-scoring yeses land above 0.5" do
      fit = described_class.fit_platt(low_but_ranked)
      calibrated = low_but_ranked.map { |p, yes| [described_class.apply_platt(p, fit), yes] }

      expect(accuracy(low_but_ranked)).to eq(8)
      expect(accuracy(calibrated)).to eq(12)
      expect(fit[:shift]).to be > 0
    end

    it "stays finite on perfectly separated examples" do
      fit = described_class.fit_platt(low_but_ranked)
      expect(fit.values).to all(be_finite)
      expect(described_class.apply_platt(0.56, fit)).to be < 1.0
    end

    it "recovers a known shift" do
      rng = Random.new(1)
      examples = Array.new(2000) do
        x = rng.rand(-4.0..4.0)
        truth = described_class.sigmoid(x + 1.5)            # true relationship
        [described_class.sigmoid(x), rng.rand < truth]       # model reports sigmoid(x)
      end
      fit = described_class.fit_platt(examples)
      expect(fit[:scale]).to be_within(0.2).of(1.0)
      expect(fit[:shift]).to be_within(0.3).of(1.5)
    end

    it "is the identity for scale 1, shift 0" do
      expect(described_class.apply_platt(0.3, scale: 1.0, shift: 0.0)).to be_within(1e-9).of(0.3)
    end

    it "handles probabilities of exactly 0 and 1" do
      expect(described_class.apply_platt(0.0, scale: 1.0, shift: 2.0)).to be_between(0.0, 1.0)
      expect(described_class.apply_platt(1.0, scale: 1.0, shift: -2.0)).to be_between(0.0, 1.0)
    end

    it "needs examples of both answers" do
      expect { described_class.fit_platt([[0.2, true], [0.9, true]]) }.to raise_error(ArgumentError, /both answers/)
    end
  end
end
