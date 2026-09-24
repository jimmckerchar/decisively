module Decisively
  # A typed answer with probabilities. No prose, nothing to parse.
  Decision = Struct.new(:type, :value, :confidence, :distribution, :latency_ms, keyword_init: true) do
    def confident?(threshold = 0.7) = confidence.to_f >= threshold
    def to_s = value.to_s
  end
end
