# End-to-end checks against the real default model, which the other specs stub out.
# Slow, and the first run downloads the model: LAYAR_REAL_MODEL=1 bundle exec rspec spec/real_model_spec.rb
RSpec.describe "the default model", :real_model do
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
