# Quick Start Guide

## Prerequisites Setup

### 1. Install Qdrant (Vector Database)

**Option A: Using Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Using Docker Compose**
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

Then run:
```bash
docker-compose up -d
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-key-here
OPENWEATHERMAP_API_KEY=your-openweathermap-key-here
LANGSMITH_API_KEY=your-langsmith-key-here
LANGSMITH_PROJECT=ai-assignment
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 4. Verify Setup

Run the setup check script:
```bash
python setup_check.py
```

## Running the Application

### Start Streamlit UI

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### 1. Upload a PDF Document

1. Use the sidebar in the Streamlit app
2. Click "Choose a PDF file"
3. Select your PDF document
4. Click "Process PDF"
5. Wait for confirmation that chunks have been added

### 2. Ask Questions

**Weather Questions:**
- "What's the weather in London?"
- "Temperature in New York"
- "How's the weather in Paris?"

**PDF Document Questions:**
- "What is machine learning?"
- "Explain the main concepts"
- Any question related to your uploaded PDF

The agent will automatically route your query to the appropriate service.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_weather_service.py -v
```

## Troubleshooting

### Qdrant Connection Error
- Ensure Qdrant is running: `docker ps`
- Check if port 6333 is accessible
- Verify QDRANT_HOST and QDRANT_PORT in `.env`

### API Key Errors
- Verify all API keys are set in `.env`
- Check for extra spaces or quotes
- Ensure API keys are active

### Import Errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+)

## Next Steps

1. Upload a PDF document
2. Test weather queries
3. Test RAG queries on your PDF
4. Check LangSmith dashboard for evaluation logs
5. Review test results

## Getting API Keys

### OpenAI
1. Visit https://platform.openai.com
2. Sign up / Sign in
3. Go to API Keys section
4. Create new secret key

### OpenWeatherMap
1. Visit https://openweathermap.org/api
2. Sign up for free account
3. Go to API Keys
4. Copy your API key

### LangSmith
1. Visit https://smith.langchain.com
2. Sign up for account
3. Go to Settings → API Keys
4. Create new API key

