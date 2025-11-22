from collections import Counter

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

INPUT_PATH = "data/book.epub"
OUTPUT_PATH = "data/100_book.txt"


def extract_text(epub_path):
    book = epub.read_epub(epub_path)
    texts = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def normalize_ascii(text: str) -> str:
    replacements = {
        "©": "(c)",

        # dash-ek / minuszok
        "–": "-",  # EN DASH
        "—": "-",  # EM DASH
        "‒": "-",  # FIGURE DASH
        "−": "-",  # MINUS SIGN

        # okos idézőjelek
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",

        # ellipsis
        "…": "...",

        # non-breaking space -> sima space
        "\u00A0": " ",

        # zero-width space -> töröljük
        "\u200B": "",
    }

    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text


def log_non_ascii_chars(text: str) -> None:
    non_ascii_chars = [c for c in text if ord(c) > 127]
    if not non_ascii_chars:
        print("Nincsenek nem-ASCII karakterek a normalizálás után.")
        return

    counter = Counter(non_ascii_chars)
    print("Nem-ASCII karakterek a normalizálás után (karakter | kód | darab):")
    for char, count in sorted(counter.items(), key=lambda x: ord(x[0])):
        codepoint = ord(char)
        print(f"{repr(char)}  U+{codepoint:04X}  x{count}")


def main():
    full_text = extract_text(INPUT_PATH)

    full_text = normalize_ascii(full_text)

    log_non_ascii_chars(full_text)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)


if __name__ == "__main__":
    main()
