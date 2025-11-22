import json
import os
import re
from collections import defaultdict

import spacy

WORD_CONTEXTS_PATH = "data/300_word_contexts.jsonl"
CHUNKS_PATH = "data/200_chunks.jsonl"
OUTPUT_PATH = "data/400_word_pos.jsonl"

# csak kisbetűs angol betűk
WORD_OK_RE = re.compile(r"^[a-z]+$")

# Kiszűrendő POS-ok: főnév, szimbólum, írásjel, egyéb szófaj, szóköz
BAD_POS = {"PROPN", "SYM", "PUNCT", "X", "SPACE"}

# spaCy modell
NLP = spacy.load("en_core_web_sm")


def accept_word_form(word: str) -> bool:
    """
    Elfogadjuk szónak, ha:
      - csak betűkből áll
      - legalább 3 karakter hosszú
    Semmi stopword, semmi extra okoskodás.
    """
    if len(word) < 3:
        return False
    if not WORD_OK_RE.match(word):
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


def normalize_for_match(text: str) -> str:
    """
    Ugyanaz az elv, mint a 300_word_contexts-ben:
    csak [A-Za-z] betűk maradnak, lower-case.
    Pl. 'Mrs.' -> 'mrs'
    """
    return re.sub(r"[^A-Za-z]", "", text).lower()


def collect_lemma_and_pos_from_contexts(word: str, contexts, chunks, nlp_cache):
    """
    Végigmegy a szó összes context chunkján, és összegyűjti:
      - az összes (lemma, POS) párt (lemma_pos_set),
      - POS -> azok a context ID-k, ahol így fordult elő (pos_to_contexts).

    Nincs fallback, csak tényleges előfordulás számít.
    """
    lemma_pos_set = set()
    pos_to_contexts = defaultdict(set)

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
            norm = normalize_for_match(token.text)
            if not norm:
                continue
            if norm == word:
                lemma = token.lemma_.lower()
                pos = token.pos_
                lemma_pos_set.add((lemma, pos))
                pos_to_contexts[pos].add(cid)

    return lemma_pos_set, pos_to_contexts


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    chunks = load_chunks()

    total_words = 0
    written_records = 0
    skipped_form = 0
    skipped_all_propn_or_empty = 0

    nlp_cache = {}  # chunk_id -> Doc

    with open(WORD_CONTEXTS_PATH, encoding="utf-8") as f_in, \
            open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            total_words += 1
            rec = json.loads(line)
            word = rec["word"]      # lower-case
            contexts = rec["contexts"]

            # 1) alak alapú szűrés (regex + hossz)
            if not accept_word_form(word):
                skipped_form += 1
                continue

            # 2) lemmák + POS-ok gyűjtése MINDEN context-ből
            lemma_pos_set, pos_to_contexts = collect_lemma_and_pos_from_contexts(
                word, contexts, chunks, nlp_cache
            )

            # invariáns: ha a 300_word_contexts szerint itt szerepel a szó,
            # akkor normál esetben legalább egy token-matchnek lennie kell
            assert lemma_pos_set, f"Nincs előfordulás a szóra: {word}, contexts={contexts}"

            # POS-ok szűrése
            lemma_pos_set = {
                (lemma, pos)
                for (lemma, pos) in lemma_pos_set
                if pos not in BAD_POS
            }
            for bad in BAD_POS:
                pos_to_contexts.pop(bad, None)

            # ha csak ilyen “rossz” POS-ként létezett, akkor nem kell a kimenetbe
            if not lemma_pos_set:
                skipped_all_propn_or_empty += 1
                continue

            # 3) egy sor MINDEN (lemma, POS) kombinációra
            for (lemma, pos) in sorted(lemma_pos_set):
                ctx_ids = pos_to_contexts.get(pos, set())
                ctx_list = sorted(ctx_ids)

                out_rec = {
                    "word": word,
                    "lemma": lemma,
                    "pos": pos,
                    "contexts": ctx_list,
                }

                f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written_records += 1

    print(
        f"Total words: {total_words}, "
        f"records written: {written_records}, "
        f"skipped_form: {skipped_form}, "
        f"skipped_all_propn_or_empty: {skipped_all_propn_or_empty} → {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
