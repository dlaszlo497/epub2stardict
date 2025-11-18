import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

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

def main():
    full_text = extract_text(INPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)

if __name__ == "__main__":
    main()
