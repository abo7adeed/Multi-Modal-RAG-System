from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


OUTPUT = Path("data/raw/documents/test.pdf")


def create_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4)

    x = 60
    y = 800

    lines = [
        "Multimodal RAG Test Document",
        "",
        "1. Introduction",
        "",
        "RAG retrieves relevant information from an external",
        "knowledge base before generating an answer.",
        "",
        "2. Multimodal RAG",
        "",
        "Multimodal RAG can work with:",
        "- Text",
        "- Images",
        "- Tables",
        "- Charts",
        "- PDF documents",
        "",
        "3. Product Information",
        "",
        "Product: AI Laptop",
        "Price: $1200",
        "RAM: 32 GB",
        "GPU: RTX 4070",
        "Storage: 1 TB SSD",
        "",
        "4. Test Question",
        "",
        "Which laptop has 32 GB of RAM?",
        "",
        "Answer: AI Laptop.",
    ]

    pdf.setFont("Helvetica", 12)

    for line in lines:
        pdf.drawString(x, y, line)
        y -= 18

    pdf.save()

    print(f"Created PDF: {OUTPUT}")


if __name__ == "__main__":
    create_pdf()