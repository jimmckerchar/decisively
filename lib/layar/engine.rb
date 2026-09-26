module Layar
  class Engine
    def initialize(config)
      @config     = config
      @load_lock  = Mutex.new
      @run_lock   = Mutex.new
    end

    def warm!
      laya? ? laya : pipeline
      self
    end

    # Pick one of N options. Returns a Decision with a full probability distribution.
    # `options` is an Array of labels, or a Hash of { value => description } when the value you
    # store (e.g. "feature_request") reads worse to the model than a plain description ("a feature request").
    # `template` phrases each option for the NLI backend; `question` is what the Laya backend is asked.
    def choice(input, options:, temperature: nil, template: nil, question: nil)
      options = normalize_options(options)
      raise ArgumentError, "choice needs at least 2 options" if options.size < 2
      if options.size > @config.max_options
        raise ArgumentError, "#{options.size} options; split into a coarse-to-fine hierarchy (max #{@config.max_options})"
      end

      timed(:choice) do
        raw =
          if laya?
            criteria = options.to_h { |value, label| [value, label == value ? nil : label] }
            laya_predict(input, choice: { type: :choice, instructions: question || @config.question, criteria: })[:choice]
          else
            nli = classify(input, options.values, multi_label: false, template: template || @config.hypothesis_template)
            options.to_h { |value, label| [value, nli.fetch(label, 0.0)] }
          end
        dist = Calibrator.apply(raw, temperature || @config.temperature)
        best, conf = dist.max_by { |_, p| p }
        [best, conf, dist]
      end
    end

    # Yes/no. Phrase `statement` as a concrete claim about the content, e.g. "The sender is offering a prize."
    # (the Laya backend also takes a question, e.g. "Is this message spam?").
    # Pass an Array of statements to answer true if any of them holds (the most likely one decides).
    def bool(input, statement:, temperature: nil)
      statements = Array(statement).map(&:to_s).uniq
      raise ArgumentError, "bool needs at least 1 statement" if statements.empty?

      timed(:bool) do
        p_yes = yes_probabilities(input, statements).values.max
        dist  = Calibrator.apply({ true => p_yes, false => 1.0 - p_yes }, temperature || @config.temperature)
        value = dist[true] >= 0.5
        [value, dist[value], dist]
      end
    end

    # 0.0..1.0 — how strongly the input supports `criterion`, e.g. "The customer is angry."
    def score(input, criterion:)
      timed(:score) do
        p = yes_probabilities(input, [criterion]).values.first
        [p.round(4), nil, { criterion => p }]
      end
    end

    private

    # Scores each statement independently, so the probabilities don't compete: multi-label NLI,
    # or one yes/no question per statement batched into a single Laya call.
    def yes_probabilities(input, statements)
      if laya?
        answers = laya_predict(input, statements.each_with_index.to_h { |s, i| [i, { type: :noul, instructions: s }] })
        statements.each_with_index.to_h { |s, i| [s, answers[i][true]] }
      else
        raw = classify(input, statements, multi_label: true, template: "{}")
        statements.to_h { |s| [s, raw.fetch(s, 0.0).to_f] }
      end
    end

    def laya? = @config.backend == :laya

    def laya_predict(input, questions)
      key = "layar:" + Digest::SHA256.hexdigest([:laya, @config.laya_model, input, questions].inspect)
      cached(key) { @run_lock.synchronize { laya.predict(input, questions) } }
    end

    # => { value => label shown to the model }
    def normalize_options(options)
      pairs = options.is_a?(Hash) ? options.map { |v, l| [v.to_s, l.to_s] } : options.map { |o| [o.to_s, o.to_s] }
      pairs = pairs.uniq(&:first)
      labels = pairs.map(&:last)
      if labels.uniq.size != labels.size
        raise ArgumentError, "options have duplicate descriptions: #{labels.tally.select { |_, n| n > 1 }.keys.inspect}"
      end
      pairs.to_h
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
        @pipeline ||= ZeroShot.load(@config.model)
      end
    end

    def laya
      @laya || @load_lock.synchronize do
        @laya ||= begin
          dir = @config.laya_model or
            raise Error, "backend :laya needs c.laya_model set to a directory holding an ONNX export of Laya"
          Laya.load(dir)
        end
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
