"""Write the results table in README.md from reports/*.json, between the metrics markers.

Run: .venv/bin/python -m scripts.update_readme
"""
import json
import re

from signbridge.config import REPORTS, ROOT

TASKS = [("words", "Words"), ("static", "Letters and numbers"), ("emotion", "Tone from the face")]


def table():
    rows = ["| Task | Test set | Accuracy | Macro F1 | Precision | Recall | Top 3 | Answers in (p95) |", "|---|---|---|---|---|---|---|---|"]
    for key, title in TASKS:
        f = REPORTS / f"{key}.json"
        if not f.exists():
            rows.append(f"| {title} | not trained yet | | | | | | |")
            continue
        r = json.loads(f.read_text())
        s = r["split"]
        test = f"{s['n_test']} clips, {len(s['test_signers'])} unseen signers" if "test_signers" in s else f"{s['n_test']} samples"
        top3 = f"{r['top3']:.1%}" if r.get("top3") is not None else "n/a"
        rows.append(f"| {title} | {test} | {r['accuracy']:.1%} | {r['f1_macro']:.1%} | {r['precision_macro']:.1%} | {r['recall_macro']:.1%} | {top3} | {r['latency_ms_p95']:.1f} ms |")
    return "\n".join(rows)


def main():
    readme = ROOT / "README.md"
    text = readme.read_text()
    new = re.sub(r"(<!-- metrics:start -->).*?(<!-- metrics:end -->)", lambda m: f"{m.group(1)}\n{table()}\n{m.group(2)}", text, flags=re.S)
    assert new != text or table() in text, "metrics markers not found in README.md"
    readme.write_text(new)
    print(table())


if __name__ == "__main__":
    main()
