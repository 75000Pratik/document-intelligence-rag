from pypdf import PdfReader

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

full_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print(f"\n--- Page {page_number} ---")
    print(text)

    if text:
        full_text += text + "\n"

print("\nTotal extracted characters:", len(full_text))
