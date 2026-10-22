"""Add topic units for the extended vocabulary to signbridge/curriculum.json.

The core units (intro, alphabet, numbers, the first 12 word units, sentences) are kept as they are. Every word of
artifacts/vocab_words.json that no core unit holds goes into an "ext_*" unit, and any word not named in TOPICS lands in
"More words", so the course always covers the whole vocabulary exactly once.

Run: .venv/bin/python -m scripts.build_curriculum
"""
import json

from signbridge.config import ROOT
from signbridge.vocab import word_labels

CURRICULUM = ROOT / "signbridge" / "curriculum.json"
TOPICS = [
    ("ext_moving", "Actions: getting about", "walk run sit stand drive travel arrive leave start stop bring"),
    ("ext_daily", "Actions: daily life", "cook clean wash buy sell pay study teach answer ask call"),
    ("ext_sharing", "Actions: with others", "meet visit take put show tell remember try practice enjoy"),
    ("ext_family", "Family and relationships", "grandmother grandfather husband wife son daughter cousin parents family"),
    ("ext_food", "Food and cooking", "egg cheese soup salad banana fruit vegetable meat candy juice tea sandwich hamburger"),
    ("ext_body", "The body", "head eyes nose mouth face heart stomach body"),
    ("ext_health", "Health", "medicine hospital pain hurt cough nurse"),
    ("ext_places", "Places", "church restaurant bank park city country office kitchen bathroom bedroom door window street"),
    ("ext_things", "Everyday things", "money key clothes shirt shoes hat bag ball game movie music gift"),
    ("ext_calendar", "Time and calendar", "week month year hour minute birthday monday tuesday wednesday thursday friday saturday sunday"),
    ("ext_describe", "Describing things", "big small long fast slow new old young ugly easy important dirty full empty different excited lonely bored"),
    ("ext_nature", "Weather and nature", "sun moon star rain snow wind weather tree flower fire ice warm spring summer winter"),
    ("ext_travel", "Getting around", "bus train airplane boat north south east west left right up down"),
    ("ext_language", "Language and Deaf culture", "english deaf hearing interpreter sign language student boss"),
    ("ext_connect", "Connecting words", "because with without maybe never always welcome excuse congratulations ready late early busy free true"),
]
SENTENCES = [  # (glosses, English); every gloss must be in the vocabulary
    ("tomorrow me visit grandmother", "I will visit my grandmother tomorrow."), ("you want eat egg", "Do you want to eat an egg?"),
    ("where bathroom", "Where is the bathroom?"), ("me need medicine", "I need medicine."), ("yesterday me buy shirt", "I bought a shirt yesterday."),
    ("you study english", "Do you study English?"), ("birthday tomorrow", "My birthday is tomorrow."), ("me drive car work", "I drive to work."),
    ("weather what", "What is the weather?"), ("me love music", "I love music."), ("rain tomorrow", "It will rain tomorrow."),
    ("you friend deaf", "Is your friend deaf?"), ("me hospital go", "I am going to the hospital."), ("money me need", "I need money."),
    ("door open please", "Please open the door."),
]


def main():
    labels = word_labels()
    data = json.loads(CURRICULUM.read_text())
    core = [u for u in data["units"] if not u["id"].startswith("ext_") and u["id"] not in ("more", "sentences")]
    sentences = next(u for u in data["units"] if u["id"] == "sentences")
    held = {i for u in core if u["kind"] == "word" for i in u["items"]}
    todo = [w for w in labels if w not in held]
    units, placed = [], set()
    for uid, title, words in TOPICS:
        items = [w for w in words.split() if w in todo and w not in placed]
        placed.update(items)
        if items:
            units.append({"id": uid, "title": title, "kind": "word", "items": items})
    rest = [w for w in todo if w not in placed]
    if rest:
        units.append({"id": "more", "title": "More words", "kind": "word", "items": rest})
    vocab = set(labels)
    base = [s for s in sentences["sentences"] if not s.get("extra")]
    extra = [{"gloss": g.split(), "english": e, "extra": True} for g, e in SENTENCES]
    for s in extra:
        assert set(s["gloss"]) <= vocab, f"sentence uses a word outside the vocabulary: {s}"
    sentences = {**sentences, "sentences": base + extra}
    data["units"] = core + units + [sentences]
    CURRICULUM.write_text(json.dumps(data, indent=1))
    counts = sum(len(u["items"]) for u in data["units"] if u["kind"] == "word")
    print(f"{len(data['units'])} units, {counts} words ({len(labels)} in the vocabulary), {len(base) + len(extra)} sentences")


if __name__ == "__main__":
    main()
