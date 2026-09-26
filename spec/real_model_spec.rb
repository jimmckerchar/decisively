# End-to-end checks against the real default model, which the other specs stub out.
# Slow, and the first run downloads the model: LAYAR_REAL_MODEL=1 bundle exec rspec spec/real_model_spec.rb
RSpec.describe "the default model", :real_model do
  # Load the model once: spec_helper resets Layar.engine after each example, and every
  # reload holds another copy of the weights (~2.4 GB for bart-large) until GC.
  before(:all) { @engine = Layar::Engine.new(Layar::Config.new) }
  before { Layar.engine = @engine }

  spam_signs = ["The sender is offering a prize.", "The message asks you to click a link."]

  it "routes a double charge to billing" do
    d = Layar.choice("I was charged twice for my subscription this month",
                     options: { "billing" => "billing", "bug" => "a bug",
                                "feature_request" => "a feature request", "account" => "account access" })
    expect(d.value).to eq("billing")
  end

  it "flags prize spam" do
    expect(Layar.bool("CONGRATS!!! You won a free cruise, click here", statement: spam_signs).value).to be(true)
  end

  it "does not flag an ordinary request" do
    expect(Layar.bool("Hi, can you send me the invoice for last month?", statement: spam_signs).value).to be(false)
  end

  it "scores an angry message high and a happy one low" do
    angry = Layar.score("This is the third time I've asked. Fix it now.", criterion: "The customer is angry.")
    happy = Layar.score("Thanks so much, that fixed it!", criterion: "The customer is angry.")
    expect(angry.value).to be > 0.7
    expect(happy.value).to be < 0.3
  end
end

# Layar::ZeroShot re-implements Informers' zero-shot pipeline with batching; check they still agree.
# Uses the small distilbert model so both copies fit in memory alongside the default model.
RSpec.describe Layar::ZeroShot, :real_model do
  model = "Xenova/distilbert-base-uncased-mnli"
  text  = "I was charged twice for my subscription this month"

  before(:all) do
    @ours   = described_class.load(model)
    @theirs = Informers.pipeline("zero-shot-classification", model)
  end

  {
    "single-label" => [["billing", "a bug", "a feature request", "account access"], false, "This example is about {}."],
    "multi-label"  => [["The customer was overcharged.", "The customer is happy."], true, "{}"],
    "one label"    => [["The customer was overcharged."], false, "{}"],
  }.each do |name, (labels, multi_label, template)|
    it "matches Informers' own pipeline (#{name})" do
      ours   = @ours.(text, labels, multi_label:, hypothesis_template: template)
      theirs = @theirs.(text, labels, multi_label:, hypothesis_template: template)

      expect(ours[:labels]).to eq(theirs[:labels])
      expect(ours[:scores]).to match(theirs[:scores].map { |s| be_within(1e-4).of(s) })
    end
  end
end
