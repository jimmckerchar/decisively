# Smoke test for an installed gem: no configuration, default model downloaded on first use.
#   gem install layar && ruby script/smoke.rb
# CI runs this on a fresh machine after installing the built gem, so it proves `gem install layar`
# works hands-off: dependencies from rubygems.org, model from the Hugging Face Hub.
require "layar"

failures = []
check = lambda do |label, actual, expected|
  ok = actual == expected
  failures << label unless ok
  puts "#{ok ? 'ok  ' : 'FAIL'} #{label}: #{actual.inspect}#{" (expected #{expected.inspect})" unless ok}"
end

puts "layar #{Layar::VERSION}, backend #{Layar.config.backend}, model #{Layar.config.laya_model}, ruby #{RUBY_VERSION} (#{RUBY_PLATFORM})"
t = Time.now
Layar.warm!
puts "loaded in #{(Time.now - t).round(1)}s (includes the download on a cold cache)"

options = %w[billing bug feature_request account]
check.("double charge", Layar.choice("I was charged twice for my subscription this month", options:).value, "billing")
check.("crash", Layar.choice("The app crashes every time I open settings", options:).value, "bug")

spam = "Is this message spam, phishing or a scam?"
check.("prize spam", Layar.bool("CONGRATS!!! You won a free cruise, click here", statement: spam).value, true)
check.("password reset", Layar.bool("Please reset my password, I can't log in to my account.", statement: spam).value, false)

t = Time.now
Layar.choice("Can I get a copy of last month's invoice?", options:)
puts "warm choice: #{((Time.now - t) * 1000).round} ms"

abort "#{failures.size} check(s) failed: #{failures.join(', ')}" if failures.any?
puts "all checks passed"
