require_relative "decidable"

module Decisive
  class Railtie < ::Rails::Railtie
    config.after_initialize do
      Decisive.config.cache ||= Rails.cache
    end
  end
end
