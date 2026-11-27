from pathlib import Path
import json
import re

INPUT_PATH = Path("data/pages.json")
OUTPUT_PATH = Path("data/cleaned.json")
HEADER_TEXT = "NORTHSTAR DIGITAL SOLUTIONS | SAMPLE EMPLOYEE RULEBOOK"


def clean_text(text, page_number=None):
    text = text.replace(HEADER_TEXT, "")
    if page_number is not None:
        text = text.replace(f"Page {page_number}", "")
    # Replace control characters, then collapse repeated whitespace.
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_document():
    if not INPUT_PATH.exists():
        raise FileNotFoundError("Run load_document.py first.")

    pages = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    cleaned_pages = [
        {"page": page["page"], "text": clean_text(page["text"], page["page"])}
        for page in pages
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(cleaned_pages, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Cleaned {len(cleaned_pages)} pages. Saved to {OUTPUT_PATH}.")
    return cleaned_pages


if __name__ == "__main__":
    clean_document()
