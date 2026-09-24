# Decisively

Laya/Jev-style "System 1" decisions for Ruby: state in, typed answer + probabilities out.
No text generation, no parsing. Runs locally via ONNX (the `informers` gem).

```ruby
Decisively.choice(text, options: %w[billing bug account])     # => Decision(value: "billing", confidence: 0.91, ...)
Decisively.bool(text, statement: "This message is spam.")     # => Decision(value: true, ...)
Decisively.score(text, criterion: "The customer is angry.")   # => Decision(value: 0.83, ...)
```

## Installation

```sh
bundle add decisively   # or: gem install decisively
```

## Rails

```ruby
# Gemfile
gem "decisively"

# config/initializers/decisively.rb
Decisively.configure { |c| c.max_options = 20 }
Decisively.warm! unless Rails.env.test?

# app/models/ticket.rb
class Ticket < ApplicationRecord
  include Decisively::Decidable
  decides :category, from: [:subject, :body], choices: %w[billing bug feature_request account]
end
```

Results are cached in `Rails.cache` automatically.

## Calibration

Raw zero-shot probabilities are overconfident. Fit a temperature on ~100+ labeled examples:

```ruby
Decisively.calibrate!(examples)  # => { temperature: 1.85, ece_before: 0.21, ece_after: 0.08 }
```

Persist the temperature and set `c.temperature = 1.85` in the initializer.

## How it differs from Laya

- Laya is a trained 421M decision model that scores all options in one forward pass.
  Decisively uses an off-the-shelf NLI cross-encoder, which runs one pass *per option*,
  so latency grows with option count.
- Zero-shot NLI generalizes worse than a purpose-trained decision model. For accuracy,
  fine-tune an NLI model on your labels and export it to ONNX, then set `c.model`.
- Like Laya, keep option sets small; use coarse-to-fine hierarchies for many labels.

## Development

Requires Ruby >= 3.1.

```sh
bundle install
bundle exec rspec        # or: bundle exec rake
```

Specs stub the Informers pipeline, so they run offline without downloading a model.
`Decisively::Decidable` can be used without Rails: `require "decisively/decidable"` (needs `activesupport`).
