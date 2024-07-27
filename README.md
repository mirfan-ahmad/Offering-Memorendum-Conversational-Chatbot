# Offering-Memorendum-Conversational-Chatbot

This project demonstrates the `Gemini` class, which is designed for extracting data from PDF files, creating embeddings with a FAISS vector database, and interacting with the Gemini 1.5 Pro conversational model. The class provides comprehensive methods for downloading PDFs, extracting text and tables, performing OCR on images, and utilizing AI for advanced data processing and conversational interactions.

## Features

- **PDF Processing:**
  - Download PDFs from a URL
  - Extract text and tables using `pdfplumber`
  - Perform OCR on images using `pytesseract`
- **FAISS Vector Database:**
  - Create embeddings from text chunks using Google Generative AI
  - Store embeddings in a FAISS vector database
- **Conversational AI:**
  - Use Gemini 1.5 Pro model for generating conversational responses

## Installation
To run the code in this project, you'll need to have Python installed on your system. Additionally, you'll need to install the required libraries.

```bash
pip install -r requirements.txt
```

Ensure you have set up your environment with the necessary API keys and libraries. You'll need a .env file with your Google API key
GEMINI_API_KEY=your_api_key_here

Here’s how to use the Gemini class for PDF processing and creating a FAISS vector database

```bash
from model import Gemini

pdf_url = "http://example.com/sample.pdf"
dictionary = {"some": "data"}
other = "additional parameter"
gemini = Gemini(pdf_url, dictionary, other)
pdf_text = gemini.pdf_text('local_path_to_pdf.pdf')
text_chunks = gemini.get_chunks(pdf_text)
gemini.get_vectors(text_chunks)
```
