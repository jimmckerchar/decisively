# Evaluates anger detection with Layar.bool on real, independently labelled text:
# GoEmotions (Reddit comments rated by people; google-research-datasets/go_emotions, Apache-2.0).
#
#   LAYAR_LAYA_MODELS=models/laya bundle exec ruby -Ilib script/eval/anger.rb [--nli]
#
# go_emotions_anger_ids.json lists the comments used, so results are reproducible:
#   validation: 100 angry (anger/annoyance only), 70 calm, 30 sad/worried -> choose wording, fit calibration
#   test:       200 angry, 150 calm, 100 sad/worried                       -> report
# "sad/worried" (sadness, disappointment, fear, nervousness) are the likely false alarms.
# Scoring takes several minutes per model; the dataset is cached under ~/.cache/layar.
require "json"
require "net/http"
require "fileutils"
require "layar"

WORDINGS = {
  "plain"          => "Is the writer angry or annoyed?",
  "at something"   => "Is the writer angry at someone or something?",
  "expresses"      => "Does the writer express anger, irritation or hostility?",
  "complaining"    => "Is the writer complaining angrily?",
  "not sad"        => "Is the writer irritated, frustrated or hostile, rather than sad or worried?",
}.freeze
DESCRIBED = { yes: "the writer is angry, annoyed, irritated or hostile",
              no:  "the writer is calm, curious, happy, grateful, sad or worried, but not angry" }.freeze

def fetch_split(split)
  cache = File.expand_path("~/.cache/layar/go_emotions/#{split}.json")
  return JSON.parse(File.read(cache)) if File.exist?(cache)

  rows, offset = [], 0
  loop do
    uri = URI("https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/go_emotions" \
              "&config=simplified&split=#{split}&offset=#{offset}&length=100")
    page = nil
    8.times do |attempt|
      res = Net::HTTP.get_response(uri)
      break page = JSON.parse(res.body) if res.is_a?(Net::HTTPSuccess)
      sleep 3 * (attempt + 1)
    end
    raise "download failed at #{split} offset #{offset}" unless page

    batch = page["rows"].map { |r| r["row"].slice("id", "text") }
    rows.concat(batch)
    offset += batch.size
    break if batch.size < 100 || offset >= page["num_rows_total"]
  end
  FileUtils.mkdir_p(File.dirname(cache))
  File.write(cache, JSON.dump(rows))
  rows
end

ids = JSON.parse(File.read(File.join(__dir__, "go_emotions_anger_ids.json")))["splits"]
sets = ids.to_h do |split, groups|
  text = fetch_split(split).to_h { |r| [r["id"], r["text"]] }
  [split, groups.flat_map { |group, list| list.map { |id| [text.fetch(id), group] } }]
end

def stats(rows)
  by = rows.group_by(&:last)
  rate = ->(g) { by[g].count { |p, _| p >= 0.5 } / by[g].size.to_f }
  { accuracy: rows.count { |p, g| (p >= 0.5) == (g == "angry") } / rows.size.to_f,
    recall: rate.("angry"), false_alarms_calm: rate.("calm"), false_alarms_sad: rate.("hard") }
end

# Fraction of (angry, not angry) pairs where the angry one scores higher: threshold-free ranking quality.
def auc(rows)
  pos = rows.select { |_, g| g == "angry" }.map(&:first)
  neg = rows.reject { |_, g| g == "angry" }.map(&:first).sort
  pos.sum { |x| neg.bsearch_index { |y| y >= x } || neg.size }.fdiv(pos.size * neg.size)
end

def pct(stats) = stats.transform_values { |v| format("%.1f%%", v * 100) }.map { |k, v| "#{k} #{v}" }.join("  ")

def report(name, variants, sets)
  puts "== #{name}"
  scored = variants.map do |label, score|
    val  = sets["validation"].map { |t, g| [score.(t), g] }
    test = sets["test"].map { |t, g| [score.(t), g] }
    printf "   %-14s validation AUC %.3f   test AUC %.3f\n", label, auc(val), auc(test)
    [label, val, test]
  end
  label, val, test = scored.max_by { |_, v, _| auc(v) }
  fit = Layar::Calibrator.fit_platt(val.map { |p, g| [p, g == "angry"] })
  puts "   chosen on validation: #{label}"
  puts "     raw        #{pct(stats(test))}"
  puts "     calibrated #{pct(stats(test.map { |p, g| [Layar::Calibrator.apply_platt(p, fit), g] }))}"
end

texts = sets.values.flatten(1).map(&:first).uniq
models = ENV["LAYAR_LAYA_MODELS"] ? %w[multilingual english].select { |c| File.directory?(File.join(ENV["LAYAR_LAYA_MODELS"], c)) } : []
abort "set LAYAR_LAYA_MODELS and/or pass --nli" if models.empty? && !ARGV.include?("--nli")

models.each do |checkpoint|
  laya = Layar::Laya.load(File.join(ENV["LAYAR_LAYA_MODELS"], checkpoint))
  questions = WORDINGS.to_h { |label, q| [label, { type: :noul, instructions: q }] }
  questions["plain + yes:/no:"] = { type: :noul, instructions: WORDINGS["plain"],
                                    criteria: { true => DESCRIBED[:yes], false => DESCRIBED[:no] } }
  scores = texts.to_h { |t| [t, laya.predict(t, questions)] }
  report("Laya #{checkpoint}", questions.keys.to_h { |l| [l, ->(t) { scores[t][l][true] }] }, sets)
  laya = nil
  GC.start
end

if ARGV.include?("--nli")
  Layar.warm!
  scores = texts.to_h { |t| [t, Layar.score(t, criterion: "The writer is angry or annoyed.").value] }
  report("NLI #{Layar.config.model}", { "plain" => ->(t) { scores[t] } }, sets)
end
