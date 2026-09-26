require "layar/decidable"

RSpec.describe Layar::Decidable do
  let(:model_class) do
    Class.new do
      include Layar::Decidable
      attr_accessor :subject, :body, :category, :priority

      def self.priorities = { "low" => 0, "high" => 1 }

      decides :category, from: %i[subject body], choices: %w[billing bug]
      decides :priority, from: :body, choices: -> { self.class.priorities.keys },
                         min_confidence: 0.6, fallback: "normal"
    end
  end

  let(:record) { model_class.new.tap { |r| r.subject = "Charged twice"; r.body = "Please refund" } }

  def decision(value, confidence)
    Layar::Decision.new(type: :choice, value:, confidence:, distribution: {})
  end

  it "joins the source fields and assigns the chosen value" do
    allow(Layar).to receive(:choice).and_return(decision("billing", 0.9))

    d = record.decide_category

    expect(Layar).to have_received(:choice).with("Charged twice\n\nPlease refund", options: %w[billing bug])
    expect(record.category).to eq("billing")
    expect(d.value).to eq("billing")
  end

  it "skips nil source fields" do
    record.subject = nil
    allow(Layar).to receive(:choice).and_return(decision("bug", 0.9))
    record.decide_category
    expect(Layar).to have_received(:choice).with("Please refund", options: anything)
  end

  it "evaluates lambda choices in the instance's context" do
    allow(Layar).to receive(:choice).and_return(decision("high", 0.9))
    record.decide_priority
    expect(Layar).to have_received(:choice).with("Please refund", options: %w[low high])
    expect(record.priority).to eq("high")
  end

  it "passes Hash choices through and assigns the value, not the description" do
    model_class.decides :category, from: :body, choices: { "feature_request" => "a feature request", "bug" => "a bug" }
    allow(Layar).to receive(:choice).and_return(decision("feature_request", 0.9))
    record.decide_category
    expect(Layar).to have_received(:choice)
      .with("Please refund", options: { "feature_request" => "a feature request", "bug" => "a bug" })
    expect(record.category).to eq("feature_request")
  end

  it "assigns the fallback below min_confidence but still returns the decision" do
    allow(Layar).to receive(:choice).and_return(decision("high", 0.5))
    d = record.decide_priority
    expect(record.priority).to eq("normal")
    expect(d.value).to eq("high")
  end
end
