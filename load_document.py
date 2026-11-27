from pathlib import Path
import json
from pypdf import PdfReader

PDF_PATH = Path("documents/northstar_company_rulebook.pdf")
OUTPUT_PATH = Path("data/pages.json")


def load_document():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}. Add your PDF with this exact name."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(PDF_PATH))

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": page_number, "text": text})

    if not any(item["text"].strip() for item in pages):
        raise ValueError(
            "No selectable text was extracted. This may be a scanned PDF; use OCR first."
        )

    OUTPUT_PATH.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Loaded {len(pages)} pages. Saved to {OUTPUT_PATH}.")
    return pages


if __name__ == "__main__":
    load_document()
