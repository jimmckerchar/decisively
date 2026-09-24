module Raya
  # Temperature scaling, the same post-hoc trick Laya uses to get its ECE down.
  module Calibrator
    module_function

    def apply(dist, t)
      return dist if t.nil? || t == 1.0
      logs = dist.transform_values { |p| Math.log([p, 1e-12].max) / t }
      max  = logs.values.max
      exps = logs.transform_values { |l| Math.exp(l - max) }
      sum  = exps.values.sum
      exps.transform_values { |e| e / sum }
    end

    # examples: [[distribution_hash, gold_key], ...]
    def fit(examples, grid: (0.25..5.0).step(0.05))
      grid.min_by { |t| nll(examples, t) }.round(2)
    end

    def nll(examples, t)
      examples.sum { |dist, gold| -Math.log([apply(dist, t)[gold].to_f, 1e-12].max) } / examples.size
    end

    # Expected calibration error: how far confidence is from actual accuracy.
    def ece(examples, bins: 10)
      buckets = Array.new(bins) { [] }
      examples.each do |dist, gold|
        label, conf = dist.max_by { |_, p| p }
        buckets[[(conf * bins).floor, bins - 1].min] << [conf, label == gold ? 1.0 : 0.0]
      end
      n = examples.size.to_f
      buckets.sum do |b|
        next 0.0 if b.empty?
        (b.size / n) * ((b.sum(&:first) / b.size) - (b.sum(&:last) / b.size)).abs
      end
    end
  end
end
