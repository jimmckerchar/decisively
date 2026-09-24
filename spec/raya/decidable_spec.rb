require "raya/decidable"

RSpec.describe Raya::Decidable do
  let(:model_class) do
    Class.new do
      include Raya::Decidable
      attr_accessor :subject, :body, :category, :priority

      def self.priorities = { "low" => 0, "high" => 1 }

      decides :category, from: %i[subject body], choices: %w[billing bug]
      decides :priority, from: :body, choices: -> { self.class.priorities.keys },
                         min_confidence: 0.6, fallback: "normal"
    end
  end

  let(:record) { model_class.new.tap { |r| r.subject = "Charged twice"; r.body = "Please refund" } }

  def decision(value, confidence)
    Raya::Decision.new(type: :choice, value:, confidence:, distribution: {})
  end

  it "joins the source fields and assigns the chosen value" do
    allow(Raya).to receive(:choice).and_return(decision("billing", 0.9))

    d = record.decide_category

    expect(Raya).to have_received(:choice).with("Charged twice\n\nPlease refund", options: %w[billing bug])
    expect(record.category).to eq("billing")
    expect(d.value).to eq("billing")
  end

  it "skips nil source fields" do
    record.subject = nil
    allow(Raya).to receive(:choice).and_return(decision("bug", 0.9))
    record.decide_category
    expect(Raya).to have_received(:choice).with("Please refund", options: anything)
  end

  it "evaluates lambda choices in the instance's context" do
    allow(Raya).to receive(:choice).and_return(decision("high", 0.9))
    record.decide_priority
    expect(Raya).to have_received(:choice).with("Please refund", options: %w[low high])
    expect(record.priority).to eq("high")
  end

  it "assigns the fallback below min_confidence but still returns the decision" do
    allow(Raya).to receive(:choice).and_return(decision("high", 0.5))
    d = record.decide_priority
    expect(record.priority).to eq("normal")
    expect(d.value).to eq("high")
  end
end
