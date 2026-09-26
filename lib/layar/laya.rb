require "json"
require "onnxruntime"
require "tokenizers"

module Layar
  # Runs a Laya decision model (https://huggingface.co/convaiinnovations/laya) exported to ONNX.
  #
  # Laya reads one sequence per question -- the question, every option behind its own mask
  # marker, then the input -- and scores each option at its marker. So a question costs one
  # pass however many options it has, and all questions about the same input share one
  # batched model call.
  #
  # A model directory holds model.onnx (+ model.onnx.data), tokenizer.json, tokenizer_config.json
  # and rl_agent_config.json; see .load for downloading one. Sequence building, option rendering and temperature scaling are a
  # port of laya 0.3.20 (Apache-2.0): common.build_sequence, common.render_options,
  # common.temp_bucket and Agent._decode_answers.
  class Laya
    QTYPES = { choice: 0, score: 1, noul: 2 }.freeze
    INPUT_NAMES = %w[input_ids attention_mask marker_pos marker_mask qtype].freeze
    OPTION_MAX_TOKENS = 48
    # Laya refuses fitted temperatures outside this range; below 0.5 they sharpen coin flips into certainties.
    TEMP_RANGE = 0.5..5.0

    # Hugging Face repo holding ONNX exports of each checkpoint (made with script/laya/export.py).
    HUB_REPO    = "distinctinteractive/laya-onnx"
    # Named checkpoints download from this commit, so a later upload can't change a released gem's
    # model (or invalidate calibrations fitted against it). Bump it deliberately, with a release.
    HUB_REVISION = "fcb4cae677a627400db0a50fddf9986e06e2b804"
    CHECKPOINTS = %w[multilingual english].freeze
    FILES       = %w[model.onnx model.onnx.data tokenizer.json tokenizer_config.json rl_agent_config.json].freeze

    # model: a checkpoint name ("multilingual", "english") downloaded from HUB_REPO, "owner/repo/subfolder"
    # on the Hugging Face Hub, or a local export directory. Downloads happen once, into Informers' cache.
    def self.load(model)
      dir = File.directory?(model.to_s) ? model.to_s : download(model.to_s)

      new(
        session:          OnnxRuntime::InferenceSession.new(File.join(dir, "model.onnx")),
        tokenizer:        Tokenizers.from_file(File.join(dir, "tokenizer.json")),
        special_tokens:   JSON.parse(File.read(File.join(dir, "tokenizer_config.json"))),
        config:           JSON.parse(File.read(File.join(dir, "rl_agent_config.json")))
      )
    end

    # => local directory holding FILES
    def self.download(model)
      repo, subfolder, revision =
        if CHECKPOINTS.include?(model)
          [HUB_REPO, model, HUB_REVISION]
        elsif (parts = model.split("/")).size >= 3 && parts.none?(&:empty?)
          [parts.first(2).join("/"), parts.drop(2).join("/"), "main"]
        else
          raise ArgumentError, "Laya model #{model.inspect} is not a local directory, a checkpoint " \
                               "(#{CHECKPOINTS.join(', ')}) or owner/repo/subfolder on the Hugging Face Hub"
        end

      paths = FILES.map do |file|
        Informers::Utils::Hub.get_model_file(repo, "#{subfolder}/#{file}", true, revision:,
                                             progress_callback: Informers::DEFAULT_PROGRESS_CALLBACK)
      end
      File.dirname(paths.first)
    end

    # special_tokens: tokenizer_config.json (cls_token, sep_token, mask_token, pad_token)
    # config: rl_agent_config.json (max_len, head_max_len, temperature, temperature_by_options)
    def initialize(session:, tokenizer:, special_tokens:, config:)
      @session   = session
      @tokenizer = tokenizer
      @mask      = special_tokens.fetch("mask_token")
      @cls_id, @sep_id, @mask_id, @pad_id =
        %w[cls_token sep_token mask_token pad_token].map { |t| token_id(special_tokens.fetch(t)) }
      # Identifies the checkpoint (not where it lives), so calibrations can tell english from multilingual.
      @identity     = [config["encoder"], config["model_name"], config.dig("training", "updates")].compact.join("/")
      @max_len      = config.fetch("max_len", 512)
      @head_max_len = config.fetch("head_max_len", 192)
      @temperature  = config.fetch("temperature", [1.0, 1.0, 1.0]).map { |t| clamp_temperature(t) }
      @temperature_by_options = config.fetch("temperature_by_options", {}).transform_values { |t| clamp_temperature(t) }
    end

    attr_reader :identity

    # state: a String, or a Hash/Array (sent as JSON; an Array is a conversation, truncated from the left).
    # questions: { id => { type: :choice | :noul | :score, instructions: String, criteria: ... } }
    #   choice  criteria: Array of options, or Hash of { option => description or nil }
    #   noul    criteria: optional { true: description, false: description }
    #   score   criteria: Array of level descriptions, lowest first
    # => { id => { option => probability } }, options in the order given
    #    (noul: { false => p, true => p }; score: { 0 => p, 1 => p, ... })
    def predict(state, questions)
      return {} if questions.empty?

      state_ids = encode(serialize(state).gsub(@mask, " "))
      items = questions.map do |id, q|
        type = q.fetch(:type).to_sym
        raise ArgumentError, "question #{id.inspect}: unknown type #{type.inspect}" unless QTYPES.key?(type)

        options = render_options(type, q[:criteria])
        ids, markers = build_sequence(state_ids, type, q.fetch(:instructions), options, truncate_left: state.is_a?(Array))
        if markers.size != options.size
          raise ArgumentError, "question #{id.inspect}: options exceed the #{@head_max_len}-token head budget"
        end
        { id:, type:, ids:, markers:, keys: option_keys(type, q[:criteria]) }
      end

      logits, _act_logits = @session.run(%w[logits act_logits], collate(items))

      items.each_with_index.to_h do |item, row|
        k = item[:markers].size
        t = @temperature_by_options.fetch(temp_bucket(item[:type], k), @temperature[QTYPES[item[:type]]])
        [item[:id], item[:keys].zip(softmax(logits[row].first(k).map { |z| z / t })).to_h]
      end
    end

    private

    # [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    def build_sequence(state_ids, type, instructions, options, truncate_left:)
      head_ids = encode("#{type} question: #{instructions.to_s.gsub(@mask, ' ')}")
      opt_ids = options.map { |o| [@mask_id] + encode(" " + o.gsub(@mask, " ")).first(OPTION_MAX_TOKENS) }

      opt_budget = @head_max_len - opt_ids.sum(&:size)
      if opt_budget < 16
        per = [4, (@head_max_len - 16) / [1, opt_ids.size].max].max
        opt_ids = opt_ids.map { |o| o.first(per) }
        opt_budget = @head_max_len - opt_ids.sum(&:size)
      end

      ids = [@cls_id] + head_ids.first([8, opt_budget].max) + [@sep_id]
      markers = opt_ids.map { |o| ids.size.tap { ids.concat(o) } }
      ids << @sep_id

      room = [0, @max_len - ids.size - 1].max
      state_part = truncate_left ? state_ids.drop([0, state_ids.size - room].max) : state_ids.first(room)
      [(ids + state_part + [@sep_id]).first(@max_len), markers.select { |m| m < @max_len }]
    end

    def render_options(type, criteria)
      case type
      when :choice
        raise ArgumentError, "a choice question needs criteria" if criteria.nil? || criteria.empty?
        return criteria.map(&:to_s) if criteria.is_a?(Array)
        criteria.map { |k, v| v.nil? || v == "" ? k.to_s : "#{k}: #{render(v)}" }
      when :score
        raise ArgumentError, "a score question needs a list of levels" unless criteria.is_a?(Array) && criteria.any?
        criteria.each_with_index.map { |c, i| "level #{i}: #{render(c)}" }
      when :noul
        crit = (criteria || {}).transform_keys { |k| k.to_s.downcase }
        [["false", "no, the statement does not hold"], ["true", "yes, the statement holds"]].map do |key, default|
          desc = crit[key]
          "#{key}: #{desc.nil? || desc == '' ? default : render(desc)}"
        end
      end
    end

    def option_keys(type, criteria)
      case type
      when :choice then criteria.is_a?(Array) ? criteria.map(&:to_s) : criteria.keys.map(&:to_s)
      when :score  then (0...criteria.size).to_a
      when :noul   then [false, true]
      end
    end

    def collate(items)
      len  = items.map { |it| it[:ids].size }.max
      kmax = items.map { |it| it[:markers].size }.max
      {
        "input_ids"      => items.map { |it| it[:ids] + [@pad_id] * (len - it[:ids].size) },
        "attention_mask" => items.map { |it| [1] * it[:ids].size + [0] * (len - it[:ids].size) },
        "marker_pos"     => items.map { |it| it[:markers] + [0] * (kmax - it[:markers].size) },
        "marker_mask"    => items.map { |it| [true] * it[:markers].size + [false] * (kmax - it[:markers].size) },
        "qtype"          => items.map { |it| QTYPES[it[:type]] },
      }
    end

    def temp_bucket(type, k)
      size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
      "#{type}:#{size}"
    end

    def clamp_temperature(t)
      t = Float(t)
      t.finite? ? t.clamp(TEMP_RANGE) : 1.0
    rescue ArgumentError, TypeError
      1.0
    end

    def encode(text) = @tokenizer.encode(text, add_special_tokens: false).ids

    def token_id(token)
      @tokenizer.token_to_id(token) or raise ArgumentError, "tokenizer has no #{token.inspect} token"
    end

    def serialize(state) = state.is_a?(String) ? state : render(state)

    # Python's json.dumps(v, ensure_ascii=False): ", " and ": " separators, strings pass through at top level.
    def render(value, top: true)
      case value
      when String then top ? value : value.to_json
      when Hash   then "{" + value.map { |k, v| "#{k.to_s.to_json}: #{render(v, top: false)}" }.join(", ") + "}"
      when Array  then "[" + value.map { |v| render(v, top: false) }.join(", ") + "]"
      when nil    then "null"
      else value.to_json
      end
    end

    def softmax(xs)
      max  = xs.max
      exps = xs.map { |x| Math.exp(x - max) }
      sum  = exps.sum
      exps.map { |e| e / sum }
    end
  end
end
