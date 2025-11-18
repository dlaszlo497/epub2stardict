import json
import os
import re

import spacy

WORD_CONTEXTS_PATH = "data/300_word_contexts.jsonl"
CHUNKS_PATH = "data/200_chunks.jsonl"
OUTPUT_PATH = "data/400_word_pos.jsonl"

# egyszerű angol stopword lista – bővíthető
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "for", "of", "in", "on", "at", "to", "from", "by", "with",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "he", "she", "it", "they", "we", "you", "i",
    "this", "that", "these", "those",
    "as", "so", "not", "no", "yes",
    "do", "does", "did", "done",
    "have", "has", "had",
    "his", "her", "their", "our", "your", "my",
    # római számok 1–20 (lower-case, a 300-as script már lower-case-el)
    "i", "ii", "iii", "iv", "v",
    "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv",
    "xvi", "xvii", "xviii", "xix", "xx",
}

# “normál” szavak: csak betű, min. 3 karakter
WORD_OK_RE = re.compile(r"^[a-z]{3,}$")

# spaCy modell
NLP = spacy.load("en_core_web_sm")


def accept_word_form(word: str) -> bool:
    """Alak alapú szűrés: regex + stopwords."""
    if not WORD_OK_RE.match(word):
        return False
    if word in STOPWORDS:
        return False
    return True


def load_chunks():
    """chunk_id -> sentence"""
    chunks = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["id"]
            chunks[cid] = rec["sentence"]
    return chunks


def collect_lemma_and_pos_from_contexts(word: str, contexts, chunks, nlp_cache):
    """
    Végigmegy a szó összes context chunkján, minden érintett mondatot
    spaCy-vel POS-oz, és összegyűjti:
      - lemmák halmazát
      - POS tagek halmazát (PROPN-t később kiszűrjük)
    """
    lemmas = set()
    poses = set()

    for cid in contexts:
        sentence = chunks.get(cid)
        if not sentence:
            continue

        if cid in nlp_cache:
            doc = nlp_cache[cid]
        else:
            doc = NLP(sentence)
            nlp_cache[cid] = doc

        for token in doc:
            if token.text.lower() == word:
                lemma = token.lemma_.lower()
                pos = token.pos_

                # egyszerű heur: -ness → főnév
                if pos == "ADJ" and lemma.endswith("ness"):
                    pos = "NOUN"

                lemmas.add(lemma)
                poses.add(pos)

    return lemmas, poses


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    chunks = load_chunks()

    total = 0
    kept = 0
    skipped_form = 0
    skipped_all_propn_or_empty = 0

    nlp_cache = {}  # chunk_id -> Doc

    with open(WORD_CONTEXTS_PATH, encoding="utf-8") as f_in, \
            open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            total += 1
            rec = json.loads(line)
            word = rec["word"]  # lower-case
            contexts = rec["contexts"]

            # 1) alak alapú szűrés (regex + stopwords)
            if not accept_word_form(word):
                skipped_form += 1
                continue

            # 2) lemmák + POS-ok gyűjtése MINDEN context-ből
            lemma_set, pos_set = collect_lemma_and_pos_from_contexts(
                word, contexts, chunks, nlp_cache
            )

            # fallback: ha semmit nem találtunk
            if not lemma_set and not pos_set:
                doc = NLP(word)
                if doc:
                    token = doc[0]
                    lemma = token.lemma_.lower()
                    pos = token.pos_
                    if pos == "ADJ" and lemma.endswith("ness"):
                        pos = "NOUN"
                    lemma_set.add(lemma)
                    pos_set.add(pos)

            # PROPN-eket töröljük
            if "PROPN" in pos_set:
                pos_set.discard("PROPN")

            if not pos_set:
                skipped_all_propn_or_empty += 1
                continue

            # lemma: ha több van, választunk egyet
            if lemma_set:
                lemma = sorted(lemma_set)[0]
            else:
                lemma = word

            pos_list = sorted(pos_set)

            out_rec = {
                "word": word,
                "lemma": lemma,
                "pos": pos_list,  # pl. ["NOUN", "VERB"]
                "contexts": contexts,  # a végén
            }

            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"Total: {total}, kept: {kept}, "
        f"skipped_form: {skipped_form}, "
        f"skipped_all_propn_or_empty: {skipped_all_propn_or_empty} → {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
