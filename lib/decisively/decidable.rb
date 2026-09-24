require "active_support/concern"

module Decisively
  # class Ticket < ApplicationRecord
  #   include Decisively::Decidable
  #   decides :category, from: [:subject, :body], choices: %w[billing bug feature_request account]
  #   decides :priority, from: :body, choices: -> { self.class.priorities.keys }, min_confidence: 0.6, fallback: "normal"
  # end
  #
  # ticket.decide_category   # sets ticket.category, returns the Decision
  module Decidable
    extend ActiveSupport::Concern

    class_methods do
      def decides(attribute, from:, choices:, min_confidence: 0.0, fallback: nil)
        define_method("decide_#{attribute}") do
          opts = choices.respond_to?(:call) ? instance_exec(&choices) : choices
          text = Array(from).map { |f| public_send(f) }.compact.join("\n\n")
          decision = Decisively.choice(text, options: opts)
          public_send("#{attribute}=", decision.confidence >= min_confidence ? decision.value : fallback)
          decision
        end
      end
    end
  end
end
