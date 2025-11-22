import json
import os
import re
from typing import List

import phunspell  # pip install phunspell

INPUT_PATH = "data/500_word_senses.jsonl"
OUTPUT_PATH = "data/550_word_senses_bad.jsonl"  # csak a hibás sorok mennek ide

# magyar + angol betűk – minimális formai szűréshez
HU_WORD_RE = re.compile(r"^[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+$")


def tokenize_hu(text: str) -> List[str]:
    """
    Magyar glossza tokenizálása: csak betűs szavak.
    Pl. "elveszett kulcsok" -> ["elveszett", "kulcsok"]
        "fiókák, kacsák"   -> ["fiókák", "kacsák"]
    """
    return re.findall(r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]+", text)


def meaning_is_probably_ok_hu(meaning: str, speller: phunspell.Phunspell) -> bool:
    """
    Akkor jó a glossza, ha:
      - van legalább egy token
      - minden token:
          * csak betűkből áll
          * hunspell szerint helyes magyar szó

    Ha nincs token -> False.
    Ha bármelyik token hibás -> False.
    """
    tokens = tokenize_hu(meaning)
    if not tokens:
        return False

    for tok in tokens:
        tok_norm = tok.lower()

        if not HU_WORD_RE.match(tok_norm):
            return False

        if not speller.lookup(tok_norm):
            return False

    return True


def main():
    # phunspell: magyar szótár – 'hu_HU'
    speller = phunspell.Phunspell("hu_HU")

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(INPUT_PATH, encoding="utf-8") as infile:
        records = [json.loads(line) for line in infile if line.strip()]

    total = len(records)
    print(f"Loaded {total} records from {INPUT_PATH}")

    bad = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        for rec in records:
            meaning_hu = rec.get("meaning_hu", "") or ""

            is_ok = meaning_is_probably_ok_hu(meaning_hu, speller)
            if not is_ok:
                bad += 1
                word = rec.get("word", "")
                # egyszerű szólista: angol -> (hibás) magyar
                print(f"{word} -> {meaning_hu}")
                # és a teljes rekordot kiírjuk a bad fájlba
                outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Processed {total} records, wrote {bad} bad records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
