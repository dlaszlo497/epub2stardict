import json
import os
import re
from collections import defaultdict

CHUNKS_PATH = "data/200_chunks.jsonl"
OUTPUT_PATH = "data/300_word_contexts.jsonl"

WORD_RE = re.compile(r"[A-Za-z]+")


def main():
    word_contexts = defaultdict(set)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["id"]
            sentence = rec["sentence"]  # csak az aktuális mondatból dolgozunk

            words = WORD_RE.findall(sentence)
            unique_words = set(w.lower() for w in words)

            for w in unique_words:
                word_contexts[w].add(cid)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for w, ids in word_contexts.items():
            rec = {
                "word": w,
                "contexts": sorted(list(ids)),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(word_contexts)} word records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
