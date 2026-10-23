"""Glosses to English and English to glosses. Gemini when GEMINI_API_KEY is set, plain offline rules otherwise."""
import json
import os
import re
import threading

from .vocab import word_labels

TONE_STYLE = {  # from the old v3 app, plus surprised
    "neutral": "Use plain, matter-of-fact wording and normal punctuation.",
    "happy": "Make it warm, cheerful and enthusiastic.",
    "sad": "Make it subdued and sorrowful.",
    "angry": "Make it firm, frustrated and forceful.",
    "surprised": "Make it sound astonished.",
}
ASL_RULES = ("The input is a list of ASL glosses, NOT English. ASL uses topic-comment order, time signs first, no articles "
             "and no 'to be'. Tokens like 'fs:ALI' are names or words fingerspelled letter by letter. Turn them into ONE "
             "correct, natural English sentence that uses every sign and adds only the small words English needs.")
END = {"none": ".", "yesno": "?", "wh": "?"}
STOPWORDS = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "of", "do", "does", "did"}
TO_GLOSS = {"i": "me", "me": "me", "my": "my", "mine": "my"}
BIGRAMS = {("thank", "you"): "thankyou"}
SYNONYMS = {"goodbye": "bye", "hi": "hello", "hey": "hello", "thanks": "thankyou", "mom": "mother", "mum": "mother", "dad": "father",
            "grandma": "grandmother", "grandpa": "grandfather", "toilet": "bathroom", "restroom": "bathroom", "ill": "sick", "yeah": "yes"}
_client, _lock = None, threading.Lock()


def _ask(prompt):
    """One Gemini call that must return a JSON object. Raises on any failure; callers fall back to offline rules."""
    global _client
    from google import genai
    from google.genai import types
    timeout = float(os.getenv("GEMINI_TIMEOUT", "8"))
    with _lock:
        if _client is None:
            _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                   http_options=types.HttpOptions(timeout=int((timeout + 2) * 1000)))
    resp = _client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=512, response_mime_type="application/json"))
    return json.loads(resp.text)


def gloss_to_english(glosses: list[str], tone: str = "neutral", question: str = "none") -> dict:
    """glosses: display strings, fingerspelled ones as 'fs:NAME'. -> {"english", "source"}."""
    if not glosses:
        raise ValueError("no glosses")
    if os.getenv("GEMINI_API_KEY"):
        try:
            style = TONE_STYLE.get(tone, TONE_STYLE["neutral"])
            out = _ask(f"{ASL_RULES}\n\nSigns: {' '.join(glosses)}\nTone: {tone}. {style}\nQuestion type: {question} "
                       "(yesno or wh means it must end with '?').\nReply as JSON: {\"english\": \"...\"}")
            english = str(out["english"]).strip()
            if english:
                return {"english": english, "source": "gemini"}
        except Exception:
            pass
    words = [g[3:].capitalize() if g.startswith("fs:") else {"me": "I"}.get(g, g) for g in glosses]
    text = " ".join(words)
    return {"english": text[0].upper() + text[1:] + END.get(question, "."), "source": "offline"}


def _spell(text):
    return {"type": "spell", "text": re.sub(r"[^A-Z0-9]", "", text.upper())}


def _clean(tokens, vocab):
    """Trust boundary for model output: unknown signs become fingerspelling, spell text is A-Z and 0-9 only."""
    out = []
    for t in tokens:
        if t.get("type") == "sign" and str(t.get("label", "")).lower() in vocab:
            out.append({"type": "sign", "label": t["label"].lower()})
        elif t.get("type") in ("sign", "spell"):
            out.append(_spell(str(t.get("text") or t.get("label") or "")))
    return [t for t in out if t["type"] == "sign" or t["text"]]


def _lookup(w, vocab):
    """Map a spoken word to a vocabulary label: synonyms first, then plural and verb endings."""
    w = SYNONYMS.get(w, w)
    for stem in (w, w[:-1] if w.endswith("s") else "", w[:-2] if w.endswith(("es", "ed")) else "", w[:-3] if w.endswith("ing") else ""):
        if stem in vocab:
            return stem
    return w


def english_to_gloss(text: str) -> dict:
    """-> {"tokens": [{"type": "sign", "label"} | {"type": "spell", "text"}], "source"}."""
    vocab = set(word_labels())
    if os.getenv("GEMINI_API_KEY"):
        try:
            out = _ask("Convert the English to ASL gloss order using ONLY these sign labels: " + ", ".join(sorted(vocab)) +
                       ". Anything not in that list (names, places, other words) must be fingerspelled. Reply as JSON: "
                       '{"tokens": [{"type": "sign", "label": "..."} or {"type": "spell", "text": "..."}]}.\n\nEnglish: ' + text)
            tokens = _clean(out["tokens"], vocab)
            if tokens:
                return {"tokens": tokens, "source": "gemini"}
        except Exception:
            pass
    words = re.findall(r"[a-z0-9']+", text.lower())
    tokens, i = [], 0
    while i < len(words):
        pair = tuple(words[i:i + 2])
        w = BIGRAMS.get(pair) or words[i]
        i += 2 if pair in BIGRAMS else 1
        if w in STOPWORDS:
            continue
        w = _lookup(TO_GLOSS.get(w, w), vocab)
        tokens.append({"type": "sign", "label": w} if w in vocab else _spell(w))
    return {"tokens": [t for t in tokens if t["type"] == "sign" or t["text"]], "source": "offline"}
