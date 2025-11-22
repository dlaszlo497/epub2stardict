import json
import struct
import datetime
import subprocess
from pathlib import Path
from functools import cmp_to_key

# --- KONSTANSOK ---

# Bemeneti JSONL fájlok és a hozzájuk tartozó modellnevek
INPUT_SOURCES = [
    (Path("data/500_word_senses_openai.jsonl"), "GPT-5-mini"),
    (Path("data/500_word_senses_gemma3.jsonl"), "gemma3:27b"),
]

# Modell-prioritás, ha ugyanarra a szóra több modellből is van bejegyzés
MODEL_PRIORITY = ["GPT-5-mini", "gemma3:27b"]

OUTPUT_DIR = Path("data/eng-hun-dict")
DICT_BASENAME = "eng-hun"  # fájlok alapneve: eng-hun.ifo / eng-hun.idx / eng-hun.dict(.dz)

BOOKNAME = "English-Hungarian dictionary"
DESCRIPTION = "English-Hungarian dictionary built from George Orwell - Animal Farm word list."

VERSION = "2.4.2"
SAME_TYPE_SEQUENCE = "x"   # mint a működő eng-hung szótárban
ENCODING = "UTF-8"
LANG = "en-hu"


# --- StarDict-féle strcmp ---

def _ascii_lower_bytes(b: bytes) -> bytes:
    """
    g_ascii_tolower byte-szinten: csak 'A'-'Z' -> 'a'-'z',
    minden más bájt változatlan marad.
    """
    arr = bytearray(b)
    for i, ch in enumerate(arr):
        if 65 <= ch <= 90:  # 'A'..'Z'
            arr[i] = ch + 32  # 'a'..'z'
    return bytes(arr)


def stardict_strcmp(s1: str, s2: str) -> int:
    """
    Pythonos megfelelője a StarDict specifikációban leírt stardict_strcmp-nek:

        a = g_ascii_strcasecmp(s1, s2)
        if (a == 0) return strcmp(s1, s2);
        else return a;

    Itt mindent UTF-8 bájtokon értelmezünk.
    """
    b1 = s1.encode("utf-8")
    b2 = s2.encode("utf-8")

    # ASCII-case-insensitive összehasonlítás
    c1 = _ascii_lower_bytes(b1)
    c2 = _ascii_lower_bytes(b2)

    if c1 < c2:
        return -1
    if c1 > c2:
        return 1

    # Ha ASCII szerint egyenlő, akkor sima byte-szintű strcmp
    if b1 < b2:
        return -1
    if b1 > b2:
        return 1
    return 0


# --- Szótár-építés ---

def build_definition(entry: dict, seen_examples_for_word: set[str], source_label: str) -> str:
    """
    Egy konkrét modell (source_label) egy sorát alakítjuk át definíciós blokká.

    A jelentés sorában a modell neve is szerepel:
        pl. "tégla (főnév) (GPT-5-mini)"
    """

    meaning_hu = (entry.get("meaning_hu") or "").strip()
    pos_ai_hu = (entry.get("pos_hu") or "").strip()
    example_surface_en = (entry.get("example_surface_en") or "").strip()
    example_lemma_en = (entry.get("example_lemma_en") or "").strip()

    lines = []

    # első sor: jelentés (szófajjal + modellnévvel)
    if meaning_hu:
        if pos_ai_hu:
            lines.append(f"{meaning_hu} ({pos_ai_hu}) ({source_label})")
        else:
            lines.append(f"{meaning_hu} ({source_label})")
    else:
        # ha nincs magyar jelentés, fallback az angol szóra
        word = (entry.get("word") or entry.get("lemma") or "").strip()
        if word:
            if pos_ai_hu:
                lines.append(f"{word} ({pos_ai_hu}) ({source_label})")
            else:
                lines.append(f"{word} ({source_label})")

    # példamondatok – szónként deduplikálva (függetlenül a modelltől)
    if example_surface_en and example_surface_en not in seen_examples_for_word:
        lines.append(example_surface_en)
        seen_examples_for_word.add(example_surface_en)

    if example_lemma_en and example_lemma_en not in seen_examples_for_word:
        lines.append(example_lemma_en)
        seen_examples_for_word.add(example_lemma_en)

    # fallback, ha minden üres
    if not lines:
        fallback = (entry.get("word") or entry.get("lemma") or "<?>").strip()
        lines.append(fallback)

    return "\n".join(lines)


def load_entries_from_sources(sources: list[tuple[Path, str]]):
    """
    Több JSONL forrásból (különböző modellek) tölti be a szótári bejegyzéseket.

    - word_to_def_blocks: word -> list[(source_label, definition_block)]
    - word_to_seen_examples: word -> set[example_sentence] (példamondatok deduplikálásához)
    """
    word_to_def_blocks: dict[str, list[tuple[str, str]]] = {}
    word_to_seen_examples: dict[str, set[str]] = {}

    for jsonl_path, source_label in sources:
        print(f"Beolvasás: {jsonl_path} [{source_label}]")
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

                seen_examples = word_to_seen_examples.setdefault(word, set())
                definition_block = build_definition(obj, seen_examples, source_label)

                if definition_block:
                    word_to_def_blocks.setdefault(word, []).append((source_label, definition_block))

    # egy angol szón belül a szófaji / modell blokkokat üres sorral választjuk el egymástól
    entries = []

    # modell-prioritás sorrend
    priority_index = {label: i for i, label in enumerate(MODEL_PRIORITY)}

    for word in sorted(word_to_def_blocks.keys(), key=cmp_to_key(stardict_strcmp)):
        blocks_with_labels = word_to_def_blocks[word]

        # blokkok rendezése modell-prioritás szerint (GPT-5-mini elöl)
        blocks_with_labels.sort(key=lambda pair: priority_index.get(pair[0], 999))

        blocks = [block for (_label, block) in blocks_with_labels]
        full_definition = "\n\n".join(blocks)
        entries.append((word, full_definition))

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
    # Megnézzük, mely input források léteznek ténylegesen
    existing_sources = [(path, label) for (path, label) in INPUT_SOURCES if path.is_file()]

    if not existing_sources:
        paths_str = ", ".join(str(p) for (p, _lbl) in INPUT_SOURCES)
        raise SystemExit(f"Nem találok egyetlen bemeneti JSONL fájlt sem. Ellenőrizd ezeket: {paths_str}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Bemeneti források:")
    for path, label in existing_sources:
        print(f"  {path} [{label}]")

    entries = load_entries_from_sources(existing_sources)
    print(f"Szócikkek száma (ok != false, címszavak): {len(entries)}")

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
