import json
import os
import re
from collections import defaultdict

import spacy

CHUNKS_PATH = "data/200_chunks.jsonl"
OUTPUT_PATH = "data/300_word_contexts.jsonl"

# spaCy modell – ugyanaz, mint a 400-asban
NLP = spacy.load("en_core_web_sm")

# csak kisbetűs angol betűk
WORD_OK_RE = re.compile(r"^[a-z]+$")


def normalize_for_match(text: str) -> str:
    """
    csak az angol betűk maradnak, minden más kidobva.
    Pl. 'Mrs.' -> 'mrs', 'can't' -> 'cant'.
    """
    return re.sub(r"[^A-Za-z]", "", text).lower()


def main():
    word_contexts = defaultdict(set)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["id"]
            sentence = rec["sentence"]

            # mondat tokenizálása spaCy-vel
            doc = NLP(sentence)

            for token in doc:
                word = normalize_for_match(token.text)
                if not word:
                    continue

                # csak "normál" szavak: min. 3 karakter, csak betű
                if len(word) < 3:
                    continue
                if not WORD_OK_RE.match(word):
                    continue

                # context ID felvétele
                word_contexts[word].add(cid)

    # fájl kiírása
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for w, ids in word_contexts.items():
            rec = {
                "word": w,
                "contexts": sorted(ids),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(word_contexts)} word records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
