
# RAG Chat with PDF

This project implements a Retrieval-Augmented Generation (RAG)-based system that allows users to upload PDFs and query them using natural language. It responds with relevant information extracted from the uploaded PDF.

## Project Structure

```
RAG Chat with PDF/
│
├── app/
│   ├── __init__.py              # Initializes the Flask app
│   ├── indexing.py              # Handles loading and processing PDFs
│   ├── query_processing.py      # Handles question-answering using RAG
│   ├── retriever.py             # Defines different retrieval methods
│   ├── routes.py                # API routes for PDF upload and query
│   ├── utils.py                 # Utility functions for file handling
│
├── uploads/                     # Directory to store uploaded PDFs
│   └── introduction_to_apple.pdf
│
├── venv/                        # Virtual environment for the project
│
├── .env                         # Environment variables
├── .gitignore                   # Ignored files for Git version control
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── run.py                       # Entry point for running the Flask app
├── testRAG.ipynb                # Jupyter Notebook for RAG testing
└── README.md                    # Project documentation
```

## Features

- **PDF Upload**: Users can upload PDF files to the system.
- **Natural Language Query**: Users can ask questions about the content of the uploaded PDF.
- **RAG Integration**: Retrieval-Augmented Generation (RAG) is used to retrieve and generate responses based on the content.

## Prerequisites

- Python 3.x
- Virtual environment (recommended)

## Setup and Installation

Follow these steps to install and run the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/DucChung2123/PDF-chatbot-with-RAG-.git
cd RAG-Chat-with-PDF
```

### 2. Set Up Virtual Environment

Create and activate a virtual environment for the project:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

- Copy the `.env.example` to `.env` and update it if needed. This file should contain environment variables such as API keys and other configurations.

### 5. Running the Application

To run the Flask application, use:

```bash
python run.py
```

The application should now be running at `localhost:8080/`.

### 6. API Endpoints

#### Upload PDF

- **Endpoint**: `/upload`
- **Method**: `POST`
- **Form-data**: `file=<your_pdf_file>`

#### Query PDF

- **Endpoint**: `/query`
- **Method**: `POST`
- **Payload**:
```json
{
  "filename": "introduction_to_apple.pdf",
  "question": "What is the history of Apple?"
}
```

### 7. Testing the RAG Setup

You can run the `testRAG.ipynb` notebook in the root directory to test the RAG setup and experiment with different queries.

## Files and Directories

- **`indexing.py`**: Responsible for loading and processing the PDF documents.
- **`query_processing.py`**: Contains the logic for processing user questions and finding answers in the PDFs.
- **`retriever.py`**: Defines various retrievers used to fetch relevant document parts.
- **`routes.py`**: Defines the Flask API routes for uploading and querying PDFs.
- **`utils.py`**: Includes utility functions for file management.

## Dependencies

The key libraries used in this project are:

- **Flask**: For building the web API.
- **langchain**: For document retrieval and question-answering.
- **PyMuPDFLoader**: For loading PDF documents.

For the complete list of dependencies, refer to `requirements.txt`.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.
