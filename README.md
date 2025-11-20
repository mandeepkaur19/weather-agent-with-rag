# AI Assignment - LangGraph Agent with RAG and Weather API

A comprehensive AI pipeline implementation using LangChain, LangGraph, and LangSmith that demonstrates embeddings, vector databases, RAG (Retrieval-Augmented Generation), and clean coding practices.

## Features

- 🤖 **LangGraph Agent**: Intelligent agentic pipeline with decision-making capabilities
- 🌤️ **Weather API Integration**: Real-time weather data fetching using OpenWeatherMap API
- 📄 **RAG System**: Question answering from PDF documents using embeddings and vector search
- 🗄️ **Vector Database**: Qdrant integration for storing and retrieving document embeddings
- 📊 **LangSmith Evaluation**: LLM response evaluation and logging
- 🎨 **Streamlit UI**: Interactive chat interface for testing the system
- ✅ **Comprehensive Tests**: Unit tests for all major components

## Architecture

```
User Query
    ↓
LangGraph Agent (Decision Node)
    ↓
    ├── Weather Route → OpenWeatherMap API → Format Response
    └── RAG Route → Vector Store Search → LLM Processing → Response
    ↓
LangSmith Evaluation
    ↓
Response to User
```

## Prerequisites

- Python 3.8 or higher
- Qdrant vector database (running locally or remotely)
- API Keys:
  - OpenAI API Key
  - OpenWeatherMap API Key
  - LangSmith API Key (optional, for evaluation)

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd "AI Assignment"
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up Qdrant**:

   Option A: Using Docker (Recommended)
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

   Option B: Using pip
   ```bash
   pip install qdrant-client
   # Qdrant server needs to be running separately
   ```

5. **Configure environment variables**:

   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_PROJECT=ai-assignment
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

## Usage

### Running the Streamlit Application

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Upload a PDF document** (optional):
   - Use the sidebar to upload a PDF file
   - Click "Process PDF" to add it to the vector store

3. **Start chatting**:
   - Ask weather questions: "What's the weather in London?"
   - Ask questions about your PDF: "What is machine learning?"
   - The agent will automatically route your query to the appropriate service

### Running Tests

```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Project Structure

```
.
├── app.py                  # Streamlit UI application
├── agent.py                # LangGraph agent with decision node
├── config.py               # Configuration settings
├── weather_service.py      # OpenWeatherMap API integration
├── pdf_processor.py        # PDF text extraction and chunking
├── vector_store.py         # Qdrant vector database integration
├── rag_service.py          # RAG query processing
├── evaluator.py            # LangSmith evaluation
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_weather_service.py
│   ├── test_rag_service.py
│   └── test_agent.py
└── uploads/               # PDF upload directory (created automatically)
```

## Implementation Details

### LangGraph Agent

The agent uses a state graph with three main nodes:
1. **Route Node**: Receives the query
2. **Decision Logic**: Determines whether to use weather API or RAG based on query keywords
3. **Weather Node**: Handles weather queries
4. **RAG Node**: Handles document-based queries

### Weather Service

- Integrates with OpenWeatherMap API
- Extracts city names from natural language queries
- Formats weather data into readable responses

### RAG System

1. **PDF Processing**: Extracts text and chunks it using RecursiveCharacterTextSplitter
2. **Embeddings**: Uses OpenAI's `text-embedding-3-small` model
3. **Vector Storage**: Stores embeddings in Qdrant with cosine similarity
4. **Retrieval**: Searches for relevant chunks based on query similarity
5. **Generation**: Uses GPT-3.5-turbo to generate answers from retrieved context

### LangSmith Evaluation

- Logs all queries and responses
- Evaluates response quality based on:
  - Response length
  - Query-response relevance
  - Error detection
  - Route-specific checks

## Testing

The project includes comprehensive unit tests:

- `test_weather_service.py`: Tests weather API integration
- `test_rag_service.py`: Tests RAG query processing
- `test_agent.py`: Tests agent routing and decision-making

Run all tests:
```bash
pytest tests/ -v
```

## LangSmith Integration

1. **Set up LangSmith**:
   - Sign up at [LangSmith](https://smith.langchain.com)
   - Get your API key
   - Add it to your `.env` file

2. **View Logs**:
   - All queries and responses are automatically logged
   - Visit your LangSmith project dashboard to view traces
   - Check evaluation scores and feedback

3. **Screenshots**:
   - After running queries, take screenshots of your LangSmith dashboard
   - Include them in your submission

## API Keys Setup

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new secret key
5. Add to `.env` file

### OpenWeatherMap API Key
1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Navigate to API Keys section
4. Copy your API key
5. Add to `.env` file

### LangSmith API Key
1. Go to [LangSmith](https://smith.langchain.com)
2. Sign up for an account
3. Navigate to Settings → API Keys
4. Create a new API key
5. Add to `.env` file

## Troubleshooting

### Qdrant Connection Error
- Ensure Qdrant is running: `docker ps` (if using Docker)
- Check QDRANT_HOST and QDRANT_PORT in `.env`
- Verify Qdrant is accessible at the specified address

### API Key Errors
- Verify all API keys are correctly set in `.env`
- Check for extra spaces or quotes in API keys
- Ensure API keys are active and have sufficient credits

### PDF Processing Issues
- Ensure PDF files are not corrupted
- Check file permissions for the uploads directory
- Verify pypdf is correctly installed

## Evaluation Criteria Coverage

✅ **LangGraph and LangChain Integration**: Fully implemented with state graph and decision nodes  
✅ **Decision-Making**: Intelligent routing based on query analysis  
✅ **Vector Database**: Qdrant integration with embeddings storage and retrieval  
✅ **LangSmith Evaluation**: Response quality evaluation and logging  
✅ **Clean Code**: Modular architecture with separation of concerns  
✅ **Tests**: Comprehensive unit tests for all components  
✅ **UI**: Streamlit interface with chat functionality  

## Future Enhancements

- [ ] Support for multiple PDF documents
- [ ] Advanced query understanding using LLM-based routing
- [ ] Conversation history and context management
- [ ] Support for other document formats (DOCX, TXT)
- [ ] Enhanced error handling and retry logic
- [ ] Performance optimization for large documents

## License

This project is created for educational purposes as part of an AI assignment.

## Contact

For questions or issues, please refer to the repository issues section.

