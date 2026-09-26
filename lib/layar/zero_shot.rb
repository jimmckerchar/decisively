module Layar
  # Zero-shot NLI classification, equivalent to Informers' "zero-shot-classification" pipeline
  # but scoring every (input, hypothesis) pair in one batched model call instead of one call
  # per label, which lets ONNX Runtime use all CPU cores (~1.5x faster).
  #
  # An NLI cross-encoder reads input and hypothesis together, so each option still costs a
  # full pass; batching only runs those passes in parallel.
  class ZeroShot
    def self.load(model)
      new(
        Informers::AutoTokenizer.from_pretrained(model),
        # Informers' zero-shot pipeline uses full-precision weights; the 8-bit ones are badly degraded for NLI.
        Informers::AutoModelForSequenceClassification.from_pretrained(model, quantized: false)
      )
    end

    def initialize(tokenizer, model)
      @tokenizer = tokenizer
      @model     = model
      label2id   = model.config[:label2id].to_h.transform_keys { |k| k.to_s.downcase }
      @entailment_id    = label2id.fetch("entailment", 2)
      @contradiction_id = label2id["contradiction"] || label2id["not_entailment"] || 0
    end

    # Same contract as the Informers pipeline: { labels:, scores: }, highest score first.
    #   multi_label: false  softmax of the entailment logits across labels (they compete)
    #   multi_label: true   entailment vs contradiction for each label on its own
    def call(input, labels, multi_label:, hypothesis_template:)
      hypotheses = labels.map { |l| hypothesis_template.sub("{}", l) }
      inputs = @tokenizer.([input] * labels.size, text_pair: hypotheses, padding: true, truncation: true)
      logits = @model.(inputs).logits

      scores =
        if multi_label || labels.size == 1
          logits.map { |row| softmax([row[@contradiction_id], row[@entailment_id]]).last }
        else
          softmax(logits.map { |row| row[@entailment_id] })
        end

      ranked = labels.zip(scores).sort_by { |_, s| -s }
      { labels: ranked.map(&:first), scores: ranked.map(&:last) }
    end

    private

    def softmax(xs)
      max  = xs.max
      exps = xs.map { |x| Math.exp(x - max) }
      sum  = exps.sum
      exps.map { |e| e / sum }
    end
  end
end
