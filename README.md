# PDF Question Answering System

This project allows users to upload PDF files and ask questions about their content. The system extracts text from the PDF, processes it, and sets up a conversational retrieval chain to provide answers to user queries.

## Setup

1. Clone this repository:

```bash
git clone https://github.com/Aditya190803/Chat-with-PDF.git
```

2. Install dependencies using pip:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

Create a `.env` file in the root directory and add the following:

```plaintext
GOOGLE_API_KEY=your_google_api_key
```

Replace `your_google_api_key` with your actual GOOOGLE API key.

Get your [API Key](https://aistudio.google.com/app/apikey) from here  
## Usage

1. Run the script:

```bash
streamlit app.py
```

2. Upload a PDF file when prompted.

3. Ask questions about the content of the uploaded PDF.

## Features

- PDF text extraction using PyPDF2.
- Natural language processing using Langchain.
- Conversational retrieval chain setup for question answering.
- Asynchronous message handling using streamlit.

## Limitations

- Currently supports only PDF files.
- Limited to text-based content in PDFs.
