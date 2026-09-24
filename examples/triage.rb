require_relative "../lib/decisively"

Decisively.warm!  # first run downloads the model

d = Decisively.choice("I was charged twice for my subscription this month",
                    options: %w[billing bug feature_request account])
puts "#{d.value} (#{(d.confidence * 100).round}%) in #{d.latency_ms}ms"
pp d.distribution

spam = Decisively.bool("CONGRATS!!! You won a free cruise, click here", statement: "This message is spam.")
puts "spam? #{spam.value} (#{spam.confidence.round(2)})"

anger = Decisively.score("This is the third time I've asked. Fix it now.", criterion: "The customer is angry.")
puts "anger: #{anger.value}"
