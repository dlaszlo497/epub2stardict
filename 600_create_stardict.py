import json
import struct
import datetime
import subprocess
from pathlib import Path


INPUT_JSONL = Path("data/500_word_senses.jsonl")

OUTPUT_DIR = Path("data/eng-hun-dict")
DICT_BASENAME = "eng-hun"  # fájlok alapneve: eng-hun.ifo / eng-hun.idx / eng-hun.dict(.dz)

BOOKNAME = "English-Hungarian dictionary"
DESCRIPTION = "English-Hungarian dictionary built from George Orwell - Animal Farm word list."

VERSION = "2.4.2"
SAME_TYPE_SEQUENCE = "x"   # mint a működő eng-hung szótárban
ENCODING = "UTF-8"
LANG = "en-hu"


def build_definition(entry: dict) -> str:
    """
    A <k> tag UTÁNI rész, pl.:

    fejezet (főnév)
    The first chapter...
    Each chapter...
    """
    meaning_hu = (entry.get("meaning_hu") or "").strip()
    pos_ai_hu = (entry.get("pos_ai_hu") or "").strip()  # pl. "főnév", "ige"
    example_surface_en = (entry.get("example_surface_en") or "").strip()
    example_lemma_en = (entry.get("example_lemma_en") or "").strip()

    lines = []

    # első sor: jelentés (szófajjal)
    if meaning_hu:
        if pos_ai_hu:
            lines.append(f"{meaning_hu} ({pos_ai_hu})")
        else:
            lines.append(meaning_hu)
    else:
        word = (entry.get("word") or entry.get("lemma") or "").strip()
        if word:
            if pos_ai_hu:
                lines.append(f"{word} ({pos_ai_hu})")
            else:
                lines.append(word)

    # példamondatok
    if example_surface_en:
        lines.append(example_surface_en)
    if example_lemma_en:
        lines.append(example_lemma_en)

    # fallback, ha minden üres
    if not lines:
        fallback = (entry.get("word") or entry.get("lemma") or "<?>").strip()
        lines.append(fallback)

    return "\n".join(lines)


def load_entries(jsonl_path: Path):
    """
    JSONL -> (word, definition) párok.

    - ok == false sorok kimaradnak
    - ugyanaz a word csak egyszer kerül be (első előfordulás)
    - UTF-8 bájtsorrend szerint rendezve (mint a működő .idx-ben)
    """
    entries_map: dict[str, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # ami "ok": false, azt kihagyjuk
            if obj.get("ok") is False:
                continue

            word = (obj.get("word") or obj.get("lemma") or "").strip()
            if not word:
                continue

            definition = build_definition(obj)

            # csak az első előfordulást tartjuk meg
            if word not in entries_map:
                entries_map[word] = definition

    entries = sorted(
        entries_map.items(),
        key=lambda kv: kv[0].encode("utf-8")
    )
    return entries


def build_dict_and_idx(entries):
    """
    .idx rekord: word\0 + offset(>I) + size(>I)
    .dict rekord (sametypesequence=x esetén):
        <k>word</k>\n
        definíció...
    """

    dict_chunks = []
    idx_chunks = []
    offset = 0

    for word, definition in entries:
        word_bytes = word.encode("utf-8")

        # mint a működő szótárban:
        # <k>word</k>\n<definition>
        full_def_text = f"<k>{word}</k>\n{definition}"
        def_bytes = full_def_text.encode("utf-8")

        # .idx entry
        idx_chunks.append(word_bytes + b"\x00")
        idx_chunks.append(struct.pack(">I", offset))
        idx_chunks.append(struct.pack(">I", len(def_bytes)))

        # .dict entry
        dict_chunks.append(def_bytes)

        offset += len(def_bytes)

    dict_data = b"".join(dict_chunks)
    idx_data = b"".join(idx_chunks)
    return dict_data, idx_data


def write_ifo(ifo_path: Path, wordcount: int, idxfilesize: int):
    today = datetime.date.today()
    date_str = today.strftime("%Y.%m.%d")

    lines = [
        "StarDict's dict ifo file",
        f"version={VERSION}",
        f"wordcount={wordcount}",
        f"idxfilesize={idxfilesize}",
        f"bookname={BOOKNAME}",
        f"date={date_str}",
        f"sametypesequence={SAME_TYPE_SEQUENCE}",
        f"description={DESCRIPTION}",
        f"encoding={ENCODING}",
        "",
        f"lang={LANG}",
    ]
    text = "\n".join(lines) + "\n"

    with ifo_path.open("w", encoding="utf-8") as f:
        f.write(text)


def main():
    if not INPUT_JSONL.is_file():
        raise SystemExit(f"Nem találom a bemeneti JSONL fájlt: {INPUT_JSONL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Beolvasás: {INPUT_JSONL}")
    entries = load_entries(INPUT_JSONL)
    print(f"Szócikkek száma (ok != false, deduplikált címszavak): {len(entries)}")

    print("dict / idx építése...")
    dict_data, idx_data = build_dict_and_idx(entries)

    dict_path = OUTPUT_DIR / f"{DICT_BASENAME}.dict"
    idx_path = OUTPUT_DIR / f"{DICT_BASENAME}.idx"
    ifo_path = OUTPUT_DIR / f"{DICT_BASENAME}.ifo"

    # .dict nyers
    with dict_path.open("wb") as f:
        f.write(dict_data)

    # .idx
    with idx_path.open("wb") as f:
        f.write(idx_data)

    # .ifo
    write_ifo(
        ifo_path=ifo_path,
        wordcount=len(entries),
        idxfilesize=len(idx_data),
    )

    # dictzip -> .dict.dz
    try:
        print("dictzip futtatása...")
        subprocess.run(["dictzip", "-f", str(dict_path)], check=True)
    except FileNotFoundError:
        print("Figyelem: 'dictzip' nem található, .dict.dz nem készült (csak sima .dict).")
    except subprocess.CalledProcessError as e:
        print(f"dictzip hiba, exit code: {e.returncode}")

    print("Kész StarDict szótár fájlok:")
    print(f"  {ifo_path}")
    print(f"  {idx_path}")
    print(f"  {dict_path}")
    dict_dz_path = dict_path.with_suffix(".dict.dz")
    if dict_dz_path.exists():
        print(f"  {dict_dz_path}")


if __name__ == "__main__":
    main()
