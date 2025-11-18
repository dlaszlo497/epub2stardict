import json
import os
import re

INPUT_PATH = "data/100_book.txt"
OUTPUT_PATH = "data/200_chunks.jsonl"

def split_to_sentences(text):
    # sortörések -> space
    text = text.replace("\n", " ")
    # mindenféle whitespace normalizálása egyetlen space-re
    text = re.sub(r"\s+", " ", text).strip()

    # mondatokra vágás
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def build_sentence_records(sentences):
    records = []
    for i, sentence in enumerate(sentences):
        rec = {
            "id": i,
            "sentence": sentence,
        }
        records.append(rec)
    return records


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        text = f.read()

    sentences = split_to_sentences(text)
    records = build_sentence_records(sentences)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} sentences to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
