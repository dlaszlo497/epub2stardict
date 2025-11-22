import json
import os
import re

import spacy

INPUT_PATH = "data/100_book.txt"
OUTPUT_PATH = "data/200_chunks.jsonl"

NLP = spacy.blank("en")
NLP.add_pipe("sentencizer")


def split_to_sentences(text: str):
    # sortörések
    text = text.replace("\n", " ")
    # whitespace-ek -> egy space
    text = re.sub(r"\s+", " ", text).strip()

    doc = NLP(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def build_sentence_records(sentences):
    return [
        {"id": i, "sentence": sentence}
        for i, sentence in enumerate(sentences)
    ]


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        text = f.read()

    sentences = split_to_sentences(text)
    records = build_sentence_records(sentences)

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} sentences to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
