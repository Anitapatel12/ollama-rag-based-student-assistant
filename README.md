# Document Processing and Q&A Application

A Streamlit-based application for document processing, question-answering, summarization, and question paper generation using Ollama's local LLMs.

## Features

- **Document Processing**
  - Extract text from PDF and DOCX files
  - Process multiple document formats in a unified way

- **AI-Powered Features**
  - Document summarization
  - Question-Answering system
  - Question paper generation
  - Text translation between languages

- **Local LLM Support**
  - Ollama integration
  - Offline-first approach

- **Vector Database**
  - FAISS for efficient similarity search
  - Sentence Transformers for embeddings

## Prerequisites

- Python 3.8+
- pip
- Ollama installed locally (https://ollama.ai/)
  - After installation, run: `ollama serve`
  - Then run: `ollama pull gemma3`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gen_ai_3
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your document and use the interactive interface to:
   - Extract and view document text
   - Ask questions about the document
   - Generate summaries
   - Create question papers
   - Translate text between languages

## Configuration

- Set environment variables in a `.env` file (copy from `.env.example` if available)
- Configure your preferred LLM settings in the application interface

## Project Structure

```
gen_ai_3/
├── app.py              # Main application file
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── .gitignore         # Git ignore file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Streamlit for the web interface
- LangChain for LLM orchestration
- FAISS for vector similarity search
- Ollama for local LLM support
