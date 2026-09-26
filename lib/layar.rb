require "informers"
require "digest"
require_relative "layar/version"
require_relative "layar/decision"
require_relative "layar/calibrator"
require_relative "layar/zero_shot"
require_relative "layar/laya"
require_relative "layar/engine"
require_relative "layar/railtie" if defined?(Rails::Railtie)

module Layar
  class Error < StandardError; end

  class Config
    attr_accessor :model, :hypothesis_template, :laya_model, :question, :temperature, :bool_calibrations,
                  :cache, :cache_ttl, :max_options
    attr_reader :backend

    def initialize
      # Any NLI zero-shot model with ONNX weights that informers supports (bert, distilbert, roberta,
      # xlm-roberta, modernbert, bart). "Xenova/distilbert-base-uncased-mnli" is ~5x faster but less accurate.
      @model               = "Xenova/bart-large-mnli"
      @hypothesis_template = "This example is about {}."
      # :nli runs the zero-shot NLI `model` above, one pass per option.
      # :laya runs a Laya decision model (ONNX export in `laya_model`), one pass per question.
      @backend             = :nli
      @laya_model          = nil   # directory with model.onnx, tokenizer.json, tokenizer_config.json, rl_agent_config.json
      @question            = "What is this about?"   # what Laya is asked for `choice` without `question:`
      @temperature         = 1.0   # set by Layar.calibrate!
      @bool_calibrations   = {}    # statement => { scale:, shift: }, set by Layar.calibrate_bool!
      @cache               = nil   # anything with #fetch(key, expires_in:) e.g. Rails.cache
      @cache_ttl           = 3600
      @max_options         = 20
    end

    def backend=(value)
      value = value.to_sym
      raise ArgumentError, "unknown backend #{value.inspect}; use :nli or :laya" unless %i[nli laya].include?(value)
      @backend = value
    end
  end

  class << self
    attr_writer :engine

    def config = (@config ||= Config.new)
    def configure = yield(config)
    def engine = (@engine ||= Engine.new(config))
    def warm! = engine.warm!

    def choice(input, options:, **kw)     = engine.choice(input, options:, **kw)
    def bool(input, statement:, **kw)     = engine.bool(input, statement:, **kw)
    def score(input, criterion:, **kw)    = engine.score(input, criterion:, **kw)

    # examples: [{ input: "...", options: [...], answer: "..." }, ...]
    # Fits a temperature on your own labeled data, stores it, and returns before/after ECE.
    def calibrate!(examples)
      raw = examples.map do |ex|
        d = engine.choice(ex[:input], options: ex[:options], temperature: 1.0)
        [d.distribution, ex[:answer].to_s]
      end
      before = Calibrator.ece(raw)
      t = Calibrator.fit(raw)
      engine.config.temperature = t   # the running engine's config, even if Layar.engine was replaced
      after = Calibrator.ece(raw.map { |dist, gold| [Calibrator.apply(dist, t), gold] })
      { temperature: t, ece_before: before.round(3), ece_after: after.round(3) }
    end

    # Fits a cut-off for one yes/no statement on your labelled examples and stores it, so
    # Layar.bool(input, statement:) applies it from then on. Pass the same `yes:` / `no:` you will
    # use in #bool; refit if you change them or the model.
    #
    #   Layar.calibrate_bool!(examples, statement: "Is the customer angry?")
    #   examples: [{ input: "...", answer: true }, ...]  (both answers needed; 50+ of each is better)
    #   # => { scale: 1.1, shift: 3.2, fitted_for: "9f2c…", threshold: 0.05, accuracy_before: 0.67, ... }
    #
    # `threshold` is the raw probability that now maps to 0.5. Persist the fit with
    # c.bool_calibrations[statement] = { scale:, shift:, fitted_for: }. `fitted_for` records the model
    # and descriptions it was fitted with; #bool warns and skips the fit if either changes.
    def calibrate_bool!(examples, statement:, **kw)
      raw = examples.map do |ex|
        d = engine.bool(ex[:input], statement:, temperature: 1.0, calibrate: false, **kw)
        [d.distribution[true], ex[:answer] == true]
      end
      fit   = Calibrator.fit_platt(raw)
      after = raw.map { |p, yes| [Calibrator.apply_platt(p, fit), yes] }
      fit   = fit.merge(fitted_for: engine.calibration_fingerprint(yes: kw[:yes], no: kw[:no]))
      engine.config.bool_calibrations[statement] = fit
      threshold = Calibrator.sigmoid(-fit[:shift] / fit[:scale])   # raw probability that now maps to 0.5
      fit.merge(
        threshold: threshold.round(4),
        accuracy_before: bool_accuracy(raw).round(3), accuracy_after: bool_accuracy(after).round(3),
        ece_before: bool_ece(raw).round(3), ece_after: bool_ece(after).round(3)
      )
    end

    private

    def bool_accuracy(pairs) = pairs.count { |p, yes| (p >= 0.5) == yes } / pairs.size.to_f
    def bool_ece(pairs) = Calibrator.ece(pairs.map { |p, yes| [{ true => p, false => 1 - p }, yes] })
  end
end
