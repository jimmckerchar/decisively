require_relative "lib/layar/version"

Gem::Specification.new do |s|
  s.name        = "layar"
  s.version     = Layar::VERSION
  s.summary     = "Laya/Jev-style typed decisions (choice, bool, score) running locally in Ruby"
  s.description = "State in, typed answer + probabilities out. Runs Convai's Laya decision model (or a " \
                  "zero-shot NLI model) locally via ONNX Runtime, downloading it on first use, with " \
                  "calibration and a Rails integration."
  s.authors     = ["Jim McKerchar"]
  s.email       = ["jim.mckerchar@gmail.com"]
  s.homepage    = "https://github.com/jimmckerchar/layar"
  s.license     = "MIT"
  s.files       = Dir["lib/**/*.rb", "README.md", "CHANGELOG.md", "LICENSE.txt"]
  s.require_paths = ["lib"]
  s.required_ruby_version = ">= 3.3"   # informers 1.3 needs 3.3
  s.metadata["source_code_uri"] = s.homepage
  s.metadata["changelog_uri"]   = "#{s.homepage}/blob/main/CHANGELOG.md"
  s.metadata["rubygems_mfa_required"] = "true"

  s.add_dependency "informers", "~> 1.3"
  s.add_dependency "onnxruntime", "~> 0.9"   # Laya backend (also used by informers)
  s.add_dependency "tokenizers", "~> 0.6"
end
