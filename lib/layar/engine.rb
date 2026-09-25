module Layar
  class Engine
    def initialize(config)
      @config     = config
      @load_lock  = Mutex.new
      @run_lock   = Mutex.new
    end

    def warm!
      pipeline
      self
    end

    # Pick one of N options. Returns a Decision with a full probability distribution.
    def choice(input, options:, temperature: nil, template: nil)
      options = options.map(&:to_s).uniq
      raise ArgumentError, "choice needs at least 2 options" if options.size < 2
      if options.size > @config.max_options
        raise ArgumentError, "#{options.size} options; split into a coarse-to-fine hierarchy (max #{@config.max_options})"
      end

      timed(:choice) do
        raw  = classify(input, options, multi_label: false, template: template || @config.hypothesis_template)
        dist = Calibrator.apply(options.to_h { |o| [o, raw.fetch(o, 0.0)] }, temperature || @config.temperature)
        best, conf = dist.max_by { |_, p| p }
        [best, conf, dist]
      end
    end

    # Yes/no. Phrase `statement` as a claim, e.g. "This message is spam."
    def bool(input, statement:, temperature: nil)
      timed(:bool) do
        p_yes = entailment(input, statement)
        dist  = Calibrator.apply({ true => p_yes, false => 1.0 - p_yes }, temperature || @config.temperature)
        value = dist[true] >= 0.5
        [value, dist[value], dist]
      end
    end

    # 0.0..1.0 — how strongly the input supports `criterion`, e.g. "The customer is angry."
    def score(input, criterion:)
      timed(:score) do
        p = entailment(input, criterion)
        [p.round(4), nil, { criterion => p }]
      end
    end

    private

    def entailment(input, statement)
      classify(input, [statement], multi_label: true, template: "{}").values.first.to_f
    end

    def classify(input, labels, multi_label:, template:)
      key = "layar:" + Digest::SHA256.hexdigest([@config.model, input, labels, multi_label, template].inspect)
      cached(key) do
        out = @run_lock.synchronize do
          pipeline.(input.to_s, labels, multi_label: multi_label, hypothesis_template: template)
        end
        ls = out[:labels] || out["labels"]
        ss = out[:scores] || out["scores"]
        ls.zip(ss).to_h
      end
    end

    def cached(key, &block)
      return yield unless @config.cache
      @config.cache.fetch(key, expires_in: @config.cache_ttl, &block)
    end

    def pipeline
      @pipeline || @load_lock.synchronize do
        @pipeline ||= Informers.pipeline("zero-shot-classification", @config.model)
      end
    end

    def timed(type)
      t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      value, conf, dist = yield
      ms = ((Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0) * 1000).round(1)
      Decision.new(type:, value:, confidence: conf, distribution: dist, latency_ms: ms)
    end
  end
end
