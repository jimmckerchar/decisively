require_relative "decidable"

module Layar
  class Railtie < ::Rails::Railtie
    config.after_initialize do
      Layar.config.cache ||= Rails.cache
    end
  end
end
