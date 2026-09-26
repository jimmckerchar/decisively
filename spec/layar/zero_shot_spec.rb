RSpec.describe Layar::ZeroShot do
  subject(:zero_shot) { described_class.new(tokenizer, model) }

  # Records what it was asked to tokenize; the "encoding" is just the hypotheses.
  let(:tokenizer) do
    Class.new do
      attr_reader :calls

      def initialize = @calls = []

      def call(texts, text_pair:, **opts)
        @calls << { texts:, text_pair:, **opts }
        text_pair
      end
    end.new
  end

  # Returns fixed [contradiction, neutral, entailment] logits per hypothesis.
  let(:model) do
    Class.new do
      attr_reader :calls, :config

      def initialize(logits, label2id)
        @logits = logits
        @config = { label2id: }
        @calls  = 0
      end

      def call(hypotheses)
        @calls += 1
        Struct.new(:logits).new(hypotheses.map { |h| @logits.fetch(h) })
      end
    end.new(logits, label2id)
  end

  let(:label2id) { { "CONTRADICTION" => 0, "NEUTRAL" => 1, "ENTAILMENT" => 2 } }
  let(:logits) do
    {
      "It is about billing." => [-2.0, 0.0, 3.0],
      "It is about a bug."   => [1.0, 0.0, 1.0],
      "It is about spam."    => [3.0, 0.0, -1.0],
    }
  end
  let(:labels) { ["billing", "a bug", "spam"] }

  def softmax(xs) = xs.map { |x| Math.exp(x) / xs.sum { |y| Math.exp(y) } }

  it "scores every hypothesis in one batched model call" do
    zero_shot.("text", labels, multi_label: false, hypothesis_template: "It is about {}.")

    expect(model.calls).to eq(1)
    expect(tokenizer.calls).to eq([{ texts: ["text"] * 3,
                                     text_pair: ["It is about billing.", "It is about a bug.", "It is about spam."],
                                     padding: true, truncation: true }])
  end

  it "makes labels compete when single-label (softmax over entailment logits)" do
    out = zero_shot.("text", labels, multi_label: false, hypothesis_template: "It is about {}.")
    expected = softmax([3.0, 1.0, -1.0])

    expect(out[:labels]).to eq(labels)
    expect(out[:scores]).to match(expected.map { |e| be_within(1e-9).of(e) })
    expect(out[:scores].sum).to be_within(1e-9).of(1.0)
  end

  it "scores labels independently when multi-label (entailment vs contradiction)" do
    out = zero_shot.("text", labels, multi_label: true, hypothesis_template: "It is about {}.")
    scores = out[:labels].zip(out[:scores]).to_h

    expect(scores["billing"]).to be_within(1e-9).of(softmax([-2.0, 3.0]).last)
    expect(scores["a bug"]).to be_within(1e-9).of(0.5)
    expect(scores["spam"]).to be_within(1e-9).of(softmax([3.0, -1.0]).last)
  end

  it "scores a single label on its own even when not multi-label" do
    out = zero_shot.("text", ["spam"], multi_label: false, hypothesis_template: "It is about {}.")
    expect(out[:scores].first).to be_within(1e-9).of(softmax([3.0, -1.0]).last)
  end

  it "returns labels highest score first" do
    out = zero_shot.("text", ["spam", "billing", "a bug"], multi_label: false, hypothesis_template: "It is about {}.")
    expect(out[:labels]).to eq(["billing", "a bug", "spam"])
  end

  context "with a model whose label order differs" do
    let(:label2id) { { "entailment" => 0, "neutral" => 1, "contradiction" => 2 } }
    let(:logits)   { { "It is about spam." => [-1.0, 0.0, 3.0] } }

    it "reads entailment and contradiction from the model's label2id" do
      out = zero_shot.("text", ["spam"], multi_label: true, hypothesis_template: "It is about {}.")
      expect(out[:scores].first).to be_within(1e-9).of(softmax([3.0, -1.0]).last)
    end
  end

  context "with a two-way entailment / not_entailment model" do
    let(:label2id) { { "entailment" => 0, "not_entailment" => 1 } }
    let(:logits)   { { "It is about spam." => [2.0, 0.0] } }

    it "treats not_entailment as the contradiction class" do
      out = zero_shot.("text", ["spam"], multi_label: true, hypothesis_template: "It is about {}.")
      expect(out[:scores].first).to be_within(1e-9).of(softmax([0.0, 2.0]).last)
    end
  end

  describe ".load" do
    it "loads the tokenizer and full-precision model" do
      allow(Informers::AutoTokenizer).to receive(:from_pretrained).and_return(tokenizer)
      allow(Informers::AutoModelForSequenceClassification).to receive(:from_pretrained).and_return(model)

      expect(described_class.load("some/model")).to be_a(described_class)
      expect(Informers::AutoTokenizer).to have_received(:from_pretrained).with("some/model")
      expect(Informers::AutoModelForSequenceClassification)
        .to have_received(:from_pretrained).with("some/model", quantized: false)
    end
  end
end
