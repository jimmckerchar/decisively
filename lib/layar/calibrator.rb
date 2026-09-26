module Layar
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

    # Platt scaling for yes/no answers: p' = sigmoid(scale * logit(p) + shift).
    # Temperature alone only spreads or squeezes probabilities around 0.5; the shift moves the
    # cut-off, so a model that ranks correctly but scores low (0.3 for "angry", 0.0 for calm)
    # can be pulled onto the right side of 0.5.
    #
    # examples: [[p_yes, true/false], ...] => { scale:, shift: }
    # Uses Platt's smoothed targets, so perfectly separated examples still give a finite fit.
    def fit_platt(examples, iterations: 100)
      raise ArgumentError, "fit_platt needs examples of both answers" unless examples.map(&:last).uniq.size == 2

      pos = examples.count(&:last).to_f
      neg = examples.size - pos
      hi, lo = (pos + 1) / (pos + 2), 1 / (neg + 2)
      xs = examples.map { |p, _| logit(p) }
      ts = examples.map { |_, yes| yes ? hi : lo }

      # Newton's method with backtracking (Lin, Lin & Weng 2007, "A note on Platt's probabilistic
      # outputs"), starting from the base rate: plain Newton steps can overshoot to absurd fits.
      a, b = 0.0, Math.log((pos + 1) / (neg + 1))
      loss = platt_loss(xs, ts, a, b)
      iterations.times do
        g_a = g_b = 0.0
        h_aa = h_bb = 1e-12
        h_ab = 0.0
        xs.zip(ts).each do |x, t|
          p = sigmoid(a * x + b)
          w = p * (1 - p)
          g_a += (p - t) * x
          g_b += (p - t)
          h_aa += w * x * x
          h_ab += w * x
          h_bb += w
        end
        break if g_a.abs < 1e-7 && g_b.abs < 1e-7

        det = h_aa * h_bb - h_ab * h_ab
        da = -(h_bb * g_a - h_ab * g_b) / det
        db = -(h_aa * g_b - h_ab * g_a) / det
        slope = g_a * da + g_b * db

        step = 1.0
        step /= 2 until step < 1e-10 ||
                        (new_loss = platt_loss(xs, ts, a + step * da, b + step * db)) < loss + 1e-4 * step * slope
        break if step < 1e-10

        a += step * da
        b += step * db
        loss = new_loss
      end
      { scale: a.round(4), shift: b.round(4) }
    end

    # Cross-entropy of sigmoid(a*x + b) against smoothed targets, computed without overflow.
    def platt_loss(xs, ts, a, b)
      xs.zip(ts).sum do |x, t|
        z = a * x + b
        # -(t log p + (1-t) log(1-p)) with p = sigmoid(z) = log(1 + e^z) - t z
        (z > 0 ? z + Math.log(1 + Math.exp(-z)) : Math.log(1 + Math.exp(z))) - t * z
      end
    end

    def apply_platt(p, calibration)
      sigmoid(calibration.fetch(:scale) * logit(p) + calibration.fetch(:shift))
    end

    def logit(p)
      p = p.clamp(1e-6, 1 - 1e-6)
      Math.log(p / (1 - p))
    end

    def sigmoid(z) = 1.0 / (1.0 + Math.exp(-z))

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
