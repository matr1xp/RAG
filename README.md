# Web Page Content Query System

A RAG (Retrieval-Augmented Generation) application that allows users to query web page content using natural language. The system features both a command-line interface and a user-friendly Streamlit web interface.

## Features

- Load and analyze any web page content
- Split content into manageable chunks for processing
- Generate embeddings using OpenAI's embedding model
- Store and retrieve relevant content using Chroma vector database
- Query content using natural language questions
- Get AI-generated answers using GPT-4
- Interactive web interface with Streamlit
- Colorful and intuitive UI design

## Technologies Used

- **LangChain v0.2+**: Framework for building LLM applications
- **OpenAI GPT-4**: Large Language Model for generating responses
- **Chroma**: Vector database for storing and retrieving embeddings
- **Streamlit**: Web interface framework
- **BeautifulSoup4**: Web scraping and HTML parsing
- **Python 3.x**: Programming language

## Prerequisites

- Python 3.x installed
- OpenAI API key
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

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
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
3. **Embedding Generation**: Content chunks are converted to embeddings using OpenAI's embedding model
4. **Vector Storage**: Embeddings are stored in a Chroma vector database
5. **Query Processing**: User questions trigger relevant content retrieval
6. **Answer Generation**: GPT-4 generates answers based on retrieved content and user questions

## Error Handling

The application includes error handling for:
- Failed webpage loading
- API errors
- Invalid URLs
- Query processing issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here]