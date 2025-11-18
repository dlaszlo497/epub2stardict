#!/usr/bin/env python3
import json
import os
import re
import requests
import random
import time

# Bemeneti / kimeneti fájlok
CHUNKS_PATH = "data/200_chunks.jsonl"      # id + sentence
WORDS_PATH = "data/400_word_pos.jsonl"     # word, lemma, pos[], contexts[]
OUTPUT_PATH = "data/500_word_senses.jsonl" # 1 sor / szó (lemma+POS szinten)

# Max ennyi példamondatot adunk át kontextusnak egy szóhoz
MAX_EXAMPLES_PER_WORD = 5

# Könyv / korpusz leírása:
BOOK_INFO = (
    'The words come from the novel "Animal Farm" (1945) by George Orwell. '
    "The story is an allegory about totalitarianism and the Russian revolution, "
    "set on a farm run by animals."
)

# Ollama beállítások
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME_GLOSS = "gemma3:27b"

# POS angol magyarázat a prompthoz
POS_DESC_EN = {
    "NOUN": "a noun (thing, person, concept)",
    "PROPN": "a proper noun (name)",
    "VERB": "a verb (action or state)",
    "AUX": "an auxiliary verb",
    "ADJ": "an adjective (describing word)",
    "ADV": "an adverb (manner, time, etc.)",
    "PRON": "a pronoun",
    "ADP": "an adposition (preposition-like word)",
    "DET": "a determiner/article",
    "NUM": "a numeral",
    "CCONJ": "a coordinating conjunction",
    "SCONJ": "a subordinating conjunction",
    "PART": "a particle",
    "INTJ": "an interjection",
    "SYM": "a symbol",
    "PUNCT": "a punctuation mark",
}

# POS magyar nevek – a szótárhoz
POS_MAP_HU = {
    "NOUN": "főnév",
    "PROPN": "tulajdonnév",
    "VERB": "ige",
    "AUX": "segédige",
    "ADJ": "melléknév",
    "ADV": "határozószó",
    "PRON": "névmás",
    "ADP": "elöljárószó",
    "DET": "névelő",
    "NUM": "számnév",
    "CCONJ": "kötőszó",
    "SCONJ": "kötőszó",
    "PART": "partikula",
    "INTJ": "indulatszó",
    "SYM": "szimbólum",
    "PUNCT": "írásjel",
}


def load_chunks():
    """
    chunk_id -> sentence
    (chunks.jsonl-ben: {"id": ..., "sentence": ...})
    """
    chunks = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["id"]
            chunks[cid] = rec["sentence"]
    return chunks


def is_bad_gloss(gloss: str) -> bool:
    """
    Eldöntjük, hogy a glossza nyilvánvalóan rossz-e:
    - üres
    - bocsánatkérés / bizonytalankodás
    - nem betűvel kezdődik
    - gyanúsan hosszú egy "szónak"
    (több szavas kifejezést engedünk, de az első token legyen normális)
    """
    if not gloss:
        return True

    g = gloss.strip()
    if not g:
        return True

    first = g.split()[0]
    lower = first.lower()

    bad_subs = [
        "sajnálom",
        "nem tudom",
        "sorry",
        "i am sorry",
        "i'm sorry",
        "unknown",
        "nincs",
        "nem ismert",
    ]
    for b in bad_subs:
        if b in lower:
            return True

    # kezdődjön betűvel (magyar ékezeteket engedjük)
    if not re.match(r"^[a-záéíóöőúüű]", lower):
        return True

    # legyen épkézláb hossz (max ~20 karakter az ELSŐ szóra)
    if len(first) > 20:
        return True

    return False


def call_ollama(model: str, prompt: str, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }
    resp = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def format_pos_options(pos_list):
    """
    POS lista → embernek/LLM-nek szóló leírás.
    Pl.: ["ADP","ADV"] -> "ADP – ... or ADV – ..."
    """
    if not pos_list:
        return "unknown (the model must infer it from the examples)"

    parts = []
    for tag in pos_list:
        desc = POS_DESC_EN.get(tag)
        if desc:
            parts.append(f"{tag} – {desc}")
        else:
            parts.append(tag)

    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return " or ".join(parts)
    else:
        return ", ".join(parts[:-1]) + " or " + parts[-1]


def parse_pos_and_gloss(line: str):
    """
    1. sor formátuma:
      POS=<TAG>; HU=<szó vagy kifejezés>
    """
    s = line.strip()

    m_pos = re.search(r"POS\s*[:=]\s*([A-Z]+)", s)
    pos_tag = m_pos.group(1) if m_pos else None

    # HU után engedünk szóközöket is, amíg nem jön újabb ';'
    m_hu = re.search(r"HU\s*[:=]\s*([^;]+)", s)
    if m_hu:
        hu_word = m_hu.group(1).strip()
    else:
        hu_word = s.split()[0]

    return pos_tag, hu_word


def parse_example_surface_en(line: str):
    """
    2. sor: a felszíni alakos példamondat teljes szövege.
    """
    return line.strip()


def parse_example_lemma_en(line: str):
    """
    3. sor: a lemma alakos példamondat teljes szövege.
    """
    return line.strip()


def generate_hungarian_gloss_for_lemma(
        lemma: str,
        word: str,
        pos_list,
        example_sentences,
):
    """
    Egy LLM-hívás LEMMA szinten:
      - lehetséges POS-ok listája
      - néhány példamondat a könyvből

    Kimenet:
      - magyar címszó vagy rövid kifejezés (1–3 szó)
      - választott POS tag
      - 2 rövid angol példamondat:
          * example_surface_en: az eredeti word alakot használva
          * example_lemma_en: a lemma alakot használva
    """
    pos_options_desc = format_pos_options(pos_list)
    pos_tags_str = ", ".join(pos_list) if pos_list else "unknown"

    examples_block = ""
    for i, s in enumerate(example_sentences, start=1):
        examples_block += f"{i}. {s}\n"

    prompt = f"""
You are building a bilingual (English -> Hungarian) dictionary for a specific book.

Book / corpus information:
{BOOK_INFO}

I will give you:
- a base English word (lemma)
- the original surface form (lowercase)
- a list of possible parts of speech (POS) for this word in the book
- several example sentences from the book where this word appears (in various forms)

YOUR JOB:
1) Decide which POS from the list best represents the MAIN dictionary sense of this word
   in the context of these examples. If several are possible, choose the most central / typical one.
2) Infer the GENERAL DICTIONARY MEANING of the word in that sense.
3) Choose ONE common Hungarian word OR SHORT EXPRESSION (1–3 words) that best matches that meaning and POS.
4) Create TWO TOTALLY DIFFERENT SHORT ENGLISH EXAMPLE SENTENCES (not from the book) that clearly show this meaning:
   - The first must use the original surface form exactly as given.
   - The second must use the lemma form.
   - Both must use the word with the chosen POS.

Details:
- Focus on the BASE WORD (lemma), not on the specific inflected forms, when deciding meaning.
- Possible POS tags for this word in the book: {pos_tags_str}
- Description of these POS tags: {pos_options_desc}
- Do NOT reuse or quote the example sentences.
- The Hungarian expression can be multi-word if that is the most natural dictionary equivalent (e.g. "szerzői jog").
- If you are uncertain, guess the most likely general dictionary meaning.
- NEVER answer with "Sajnálom", "Nem tudom", "I am sorry", "Sorry" or any similar meta-reply.

Word (lemma): {lemma}
Surface form (lowercase): {word}

Example sentences from the book:
{examples_block}

VERY IMPORTANT OUTPUT FORMAT (EXACTLY THREE LINES, NO BULLETS, NO EXTRA TEXT):
Line 1: POS=<POS_TAG>; HU=<ONE_HUNGARIAN_WORD_OR_SHORT_EXPRESSION>
Line 2: <one short English sentence containing the surface form '{word}'>
Line 3: <one short English sentence containing the lemma '{lemma}'>

<POS_TAG> MUST be one of: {pos_tags_str}, or a reasonable guess if they are empty.
<ONE_HUNGARIAN_WORD_OR_SHORT_EXPRESSION> can be ONE or SEVERAL (1–3) Hungarian words.
"""

    raw = call_ollama(MODEL_NAME_GLOSS, prompt, temperature=0.1)

    # Ha debuggolni akarsz, ezt kommentezd vissza:
    # print("\n================ RAW RESPONSE ================")
    # print(f"WORD={word!r}, LEMMA={lemma!r}")
    # print(raw)
    # print("=============================================\n", flush=True)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    if not lines:
        raise ValueError("empty response from model")

    first_line = lines[0]
    second_line = lines[1] if len(lines) > 1 else ""
    third_line = lines[2] if len(lines) > 2 else ""

    pos_tag, hu_word = parse_pos_and_gloss(first_line)
    example_surface_en = parse_example_surface_en(second_line)
    example_lemma_en = parse_example_lemma_en(third_line)

    if is_bad_gloss(hu_word):
        raise ValueError(
            f"bad gloss from main model: {hu_word!r} (raw first line: {first_line!r})"
        )

    return hu_word, pos_tag, example_surface_en, example_lemma_en


def format_eta(seconds: int) -> str:
    """
    Másodpercből HH:MM:SS string.
    """
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading chunks (sentences)...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} sentences from {CHUNKS_PATH}")

    # beolvassuk az összes szót, hogy tudjuk a total-t
    with open(WORDS_PATH, encoding="utf-8") as f_in:
        word_recs = [json.loads(line) for line in f_in]

    total_words = len(word_recs)
    print(f"Loaded {total_words} word entries from {WORDS_PATH}")
    print(f"MAX_EXAMPLES_PER_WORD = {MAX_EXAMPLES_PER_WORD}")

    written = 0
    start_time = time.time()  # indulási idő az ETA-hoz

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for w_idx, rec in enumerate(word_recs, start=1):
            word = rec["word"]                 # felszíni forma (lowercase)
            lemma = rec.get("lemma") or word   # lemma
            pos_list = rec.get("pos") or []    # lehetséges POS-ok
            ctx_ids = rec["contexts"]          # mondat-azonosítók

            total_ctx_for_word = len(ctx_ids)

            # véletlenül kiválasztunk max N példamondatot kontextusnak
            valid_ids = [cid for cid in ctx_ids if cid in chunks]
            if not valid_ids:
                print(
                    f"[word {w_idx}/{total_words}] '{word}' – SKIP: no valid contexts found"
                )
                continue

            if MAX_EXAMPLES_PER_WORD is not None and len(valid_ids) > MAX_EXAMPLES_PER_WORD:
                selected_ids = random.sample(valid_ids, MAX_EXAMPLES_PER_WORD)
            else:
                selected_ids = list(valid_ids)

            example_sentences = [chunks[cid] for cid in selected_ids]

            print(
                f"\n=== WORD {w_idx}/{total_words}: '{word}' "
                f"(lemma='{lemma}', {len(selected_ids)}/{total_ctx_for_word} example sentences used) ===",
                flush=True,
            )

            try:
                gloss_hu, pos_ai, ex_surface, ex_lemma = generate_hungarian_gloss_for_lemma(
                    lemma=lemma,
                    word=word,
                    pos_list=pos_list,
                    example_sentences=example_sentences,
                )
                status = "OK"
            except Exception as e:
                gloss_hu = ""
                pos_ai = None
                ex_surface = ""
                ex_lemma = ""
                status = f"ERROR: {e}"

            pos_hu_list = [POS_MAP_HU[p] for p in pos_list if p in POS_MAP_HU]
            pos_ai_hu = POS_MAP_HU.get(pos_ai, "") if pos_ai else ""

            out_rec = {
                "word": word,
                "lemma": lemma,
                "pos": pos_list,                 # lehetséges POS-ok (spaCy tag-ek)
                "pos_hu": pos_hu_list,           # magyar megnevezések listája
                "pos_ai": pos_ai,                # AI által kiválasztott POS (vagy None)
                "pos_ai_hu": pos_ai_hu,          # AI POS magyarul
                "meaning_hu": gloss_hu,          # magyar alapszó vagy több szavas kifejezés
                "example_surface_en": ex_surface,  # felszíni alakos példamondat
                "example_lemma_en": ex_lemma,      # lemma-alakos példamondat
                "ok": gloss_hu != "",
            }

            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f_out.flush()
            written += 1

            # ETA számolás
            elapsed = time.time() - start_time
            avg_per_word = elapsed / w_idx
            remaining_words = total_words - w_idx
            eta_seconds = int(remaining_words * avg_per_word)
            eta_str = format_eta(eta_seconds)

            print(
                f"[word {w_idx}/{total_words}] '{word}' -> {status} "
                f"(HU='{gloss_hu}', POS_AI={pos_ai})",
                flush=True,
            )
            print(
                f"    Elapsed: {format_eta(int(elapsed))}, "
                f"ETA remaining: {eta_str}",
                flush=True,
            )

    total_elapsed = time.time() - start_time
    print(f"Done, written {written} entries to {OUTPUT_PATH}")
    print(f"Total elapsed time: {format_eta(int(total_elapsed))}")


if __name__ == "__main__":
    main()
