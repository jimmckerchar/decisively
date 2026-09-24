require "informers"
require "digest"
require_relative "decisively/version"
require_relative "decisively/decision"
require_relative "decisively/calibrator"
require_relative "decisively/engine"
require_relative "decisively/railtie" if defined?(Rails::Railtie)

module Decisively
  class Error < StandardError; end

  class Config
    attr_accessor :model, :hypothesis_template, :temperature, :cache, :cache_ttl, :max_options

    def initialize
      # Any NLI zero-shot model with ONNX weights works; swap in a multilingual one if you need it.
      @model               = "Xenova/mobilebert-uncased-mnli"
      @hypothesis_template = "This example is about {}."
      @temperature         = 1.0   # set by Decisively.calibrate!
      @cache               = nil   # anything with #fetch(key, expires_in:) e.g. Rails.cache
      @cache_ttl           = 3600
      @max_options         = 20
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
      config.temperature = t
      after = Calibrator.ece(raw.map { |dist, gold| [Calibrator.apply(dist, t), gold] })
      { temperature: t, ece_before: before.round(3), ece_after: after.round(3) }
    end
  end
end
