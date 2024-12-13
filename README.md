# Web Page Content Query System

## Project Summary

The Web Page Content Query System is an advanced application designed to facilitate the retrieval and querying of web page content using natural language. Leveraging state-of-the-art technologies such as LangChain, Ollama, and Chroma, this system provides both a command-line interface and an interactive Streamlit web interface. Users can load web pages, process their content, and ask questions to receive AI-generated answers. The system is built to handle complex queries efficiently, making it a powerful tool for information retrieval and analysis.

## Features

- Load and analyze any web page content
- Split content into manageable chunks for processing
- Generate embeddings using Ollama's local embedding model
- Store and retrieve relevant content using Chroma vector database
- Query content using natural language questions
- Get AI-generated answers using local Ollama models
- Interactive web interface with Streamlit
- Colorful and intuitive UI design

## Technologies Used

- **LangChain v0.2+**: Framework for building LLM applications
- **Ollama**: Local Large Language Model for generating responses
- **Chroma**: Vector database for storing and retrieving embeddings
- **Streamlit**: Web interface framework
- **BeautifulSoup4**: Web scraping and HTML parsing
- **Python 3.x**: Programming language

## Prerequisites

- Python 3.x installed
- Ollama installed and running locally
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running locally with the required models:
```bash
ollama pull llama3.1
```

## Usage

### Command Line Interface

Run the command-line version:
```bash
python rag_app.py
```

Follow the prompts to:
1. Enter a webpage URL
2. Ask questions about the content
3. Type 'new' to analyze a different webpage
4. Type 'quit' to exit

### Streamlit Web Interface

Run the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

The web interface provides:
- URL input field for loading web pages
- Question input for querying content
- Clear button to reset the application
- Visual feedback for successful/failed operations

## Project Structure

- `rag_app.py`: Core RAG functionality and CLI interface
- `streamlit_app.py`: Streamlit web interface
- `requirements.txt`: Project dependencies
- `chroma_db/`: Directory for vector database storage

## How It Works

1. **Web Page Loading**: The application fetches and parses web page content using WebBaseLoader
2. **Content Processing**: Text is split into chunks using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Content chunks are converted to embeddings using Ollama's local embedding model
4. **Vector Storage**: Embeddings are stored in a Chroma vector database
5. **Query Processing**: User questions trigger relevant content retrieval
6. **Answer Generation**: Ollama generates answers based on retrieved content and user questions

## Error Handling

The application includes error handling for:
- Failed webpage loading
- API errors
- Invalid URLs
- Query processing issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](https://opensource.org/license/mit).
