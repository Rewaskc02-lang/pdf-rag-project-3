from pypdf import PdfReader

reader = PdfReader("northstar_company_rulebook.pdf")
full_cleaned_text = ""

for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    text = text.replace("NORTHSTAR DIGITAL SOLUTIONS | SAMPLE EMPLOYEE RULEBOOK", "")
    text = text.replace(f"Page {page_num + 1}", "")
    
    clean_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            clean_lines.append(line)
            
    page_text = " ".join(clean_lines)
    full_cleaned_text += f"\n\n--- Page {page_num + 1} ---\n\n" + page_text

with open("cleaned_rulebook.txt", "w", encoding="utf-8") as file:
    file.write(full_cleaned_text.strip())

print("PDF cleaned successfully and saved to cleaned_rulebook.txt!")
