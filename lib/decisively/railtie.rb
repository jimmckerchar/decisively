require_relative "decidable"

module Decisively
  class Railtie < ::Rails::Railtie
    config.after_initialize do
      Decisively.config.cache ||= Rails.cache
    end
  end
end
