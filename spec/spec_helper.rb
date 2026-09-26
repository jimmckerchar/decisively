require "layar"

# Stands in for the Informers zero-shot pipeline so specs never load a model.
# `scores` maps label => probability; unknown labels score 0.0.
class FakePipeline
  attr_reader :calls

  def initialize(scores = {})
    @scores = scores
    @calls  = []
  end

  def call(input, labels, multi_label:, hypothesis_template:)
    @calls << { input:, labels:, multi_label:, hypothesis_template: }
    ranked = labels.map { |l| [l, @scores.fetch(l, 0.0)] }.sort_by { |_, s| -s }
    { labels: ranked.map(&:first), scores: ranked.map(&:last) }
  end
end

# Minimal stand-in for Rails.cache.
class MemoryCache
  attr_reader :store, :fetches

  def initialize
    @store   = {}
    @fetches = []
  end

  def fetch(key, expires_in:)
    @fetches << { key:, expires_in: }
    @store.key?(key) ? @store[key] : (@store[key] = yield)
  end
end

RSpec.configure do |config|
  config.example_status_persistence_file_path = ".rspec_status"
  config.disable_monkey_patching!
  config.expect_with(:rspec) { |c| c.syntax = :expect }

  # Specs tagged :real_model download and run the actual model; opt in with LAYAR_REAL_MODEL=1.
  config.filter_run_excluding(:real_model) unless ENV["LAYAR_REAL_MODEL"]

  config.after do
    Layar.instance_variable_set(:@config, nil)
    Layar.engine = nil
  end
end
