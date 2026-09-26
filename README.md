# Layar

Laya/Jev-style "System 1" decisions for Ruby: state in, typed answer + probabilities out.
No text generation, no parsing. Runs locally via ONNX Runtime, either on a zero-shot NLI model
(the default) or on Laya itself.

```ruby
Layar.choice(text, options: %w[billing bug account])     # => Decision(value: "billing", confidence: 0.91, ...)
Layar.bool(text, statement: "This message is spam.")     # => Decision(value: true, ...)
Layar.score(text, criterion: "The customer is angry.")   # => Decision(value: 0.83, ...)
```

## Installation

```sh
bundle add layar   # or: gem install layar
```

## Rails

```ruby
# Gemfile
gem "layar"

# config/initializers/layar.rb
Layar.configure { |c| c.max_options = 20 }
Layar.warm! unless Rails.env.test?

# app/models/ticket.rb
class Ticket < ApplicationRecord
  include Layar::Decidable
  decides :category, from: [:subject, :body], choices: %w[billing bug feature_request account]
end
```

Results are cached in `Rails.cache` automatically.

## Calibration

Raw zero-shot probabilities are overconfident. Fit a temperature on ~100+ labeled examples:

```ruby
Layar.calibrate!(examples)  # => { temperature: 1.85, ece_before: 0.21, ece_after: 0.08 }
```

Persist the temperature and set `c.temperature = 1.85` in the initializer.

## Backends

| | `:nli` (default) | `:laya` |
|---|---|---|
| Model | a zero-shot NLI model via `informers`; default `Xenova/bart-large-mnli` | a [Laya](https://huggingface.co/convaiinnovations/laya) decision model, exported to ONNX |
| Cost of a `choice` | one pass per option (batched into one call) | one pass per question, however many options |
| Setup | downloads on first use | export once with the scripts below |

```ruby
Layar.configure do |c|
  c.backend    = :laya
  c.laya_model = "/models/laya/multilingual"   # directory written by script/laya/export.py
end

Layar.choice(text, options: %w[billing bug account], question: "Which team should handle this?")
Layar.bool(text, statement: "Is this message spam, phishing or a scam?")
```

With `:laya`:
- `choice` asks `question:` (default `c.question`, "What is this about?"). Hash options
  `{ value => description }` are shown to Laya as "value: description".
- `bool` asks each statement as a yes/no question (a question or a claim both work), all in one
  batched call, and is true if any holds.
- `score` is the probability that the criterion holds.

In spot checks on a 4-core CPU (`spec/laya_real_model_spec.rb`, `spec/real_model_spec.rb`), the
multilingual checkpoint answered a 4-option `choice` in ~130 ms against ~540 ms for bart-large-mnli,
barely slowed down with 8 options, and told all 9 spam and ordinary messages apart where
bart-large-mnli managed at best 7. That is a small test set: check accuracy on your own data.

Laya ships overconfident (the multilingual checkpoint has no fitted temperatures), so run
`Layar.calibrate!` on labelled examples before gating on confidence. It is also conservative about
emotions: "This is the third time I've asked. Fix it now." scored only 0.10-0.37 for anger across
four phrasings (0.00 for a thank-you note), so compare `score` values rather than gating at 0.5.
Broad statements work as questions or claims ("This message is spam." told all 9 apart), but narrow
ones mislead it as they do NLI ("The message asks you to click a link." flags ordinary requests).

### Exporting Laya to ONNX

Laya publishes PyTorch weights only, so export a checkpoint once. Needs Python 3.10+, about 1 GB of
packages and up to ~4 GB of RAM while exporting:

```sh
python -m venv .laya && . .laya/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r script/laya/requirements.txt
python script/laya/export.py multilingual models/laya/multilingual
python script/laya/parity.py multilingual models/laya/multilingual   # ONNX Runtime vs PyTorch
```

| Checkpoint | Size | Notes |
|---|---|---|
| `multilingual` | 1.3 GB | recommended: fastest, 100+ languages, inputs up to 1,024 tokens |
| `english` | 1.7 GB | ModernBERT-large; slower on CPU |
| `typed-decisions` | ~1.7 GB | fine-tuned to Laya's own benchmark; weaker in general use (export untested) |

Laya is by Convai Innovations, licensed Apache-2.0. `Layar::Laya` ports its input formatting from
the `laya` 0.3.20 Python package.

## How it differs from Laya

- With `:nli`, Layar rephrases decisions as entailment questions for an off-the-shelf NLI model.
  It matches text on topic more than intent: "Please reset my password" looks like phishing to it.
  Fine-tune an NLI model on your labels and set `c.model`, or use `:laya`.
- With `:laya`, Layar runs Laya's model and input format, but not its Python extras (router,
  hooks, act/escalate head, per-language temperatures).
- Like Laya, keep option sets small; use coarse-to-fine hierarchies for many labels.

## Development

Requires Ruby >= 3.1.

```sh
bundle install
bundle exec rspec        # or: bundle exec rake
```

Specs stub the model, so they run offline without downloading anything. End-to-end specs
against the real model are opt-in (slow; needs ~3 GB RAM for the default model):

```sh
LAYAR_REAL_MODEL=1 bundle exec rspec spec/real_model_spec.rb
LAYAR_REAL_MODEL=1 LAYAR_LAYA_MODELS=models/laya bundle exec rspec spec/laya_real_model_spec.rb
```

The Laya specs replay `spec/fixtures/laya/*.json`, recorded from Python Laya by
`script/laya/record_fixture.py`, so the Ruby port must reproduce its token ids and probabilities.
Run the two files separately on small machines; each model holds ~2 GB while loaded.

`Layar::Decidable` can be used without Rails: `require "layar/decidable"` (needs `activesupport`).
