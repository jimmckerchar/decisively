require_relative "decidable"

module Raya
  class Railtie < ::Rails::Railtie
    config.after_initialize do
      Raya.config.cache ||= Rails.cache
    end
  end
end
