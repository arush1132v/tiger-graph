# 🔬 RAG Pipeline Comparison Hackathon Project

A comprehensive comparison dashboard for three different RAG (Retrieval-Augmented Generation) approaches, designed for hackathon demonstration.

## 🎯 Project Overview

This project demonstrates the evolution of question-answering systems:

1. **Pipeline 1: LLM-Only** (Baseline)
   - Direct LLM inference with no retrieval
   - Relies solely on model's training data
   - Worst-case baseline for comparison

2. **Pipeline 2: Vector RAG** (Current Standard)
   - Vector embeddings for semantic similarity
   - Retrieves relevant chunks from your data
   - Uses ChromaDB for vector storage

3. **Pipeline 3: GraphRAG** (Advanced)
   - Knowledge graph-based retrieval
   - Multi-hop reasoning through relationships
   - Uses TigerGraph for graph operations

## 📁 Project Structure

```
├── config.py              # Configuration management
├── ingest_data.py         # Data ingestion & vector DB creation
├── pipeline_1_llm.py      # Pipeline 1: LLM-Only
├── pipeline_2_rag.py      # Pipeline 2: Vector RAG
├── pipeline_3_graphrag.py # Pipeline 3: GraphRAG (placeholder)
├── dashboard.py           # Streamlit comparison dashboard
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd rag-pipeline-comparison

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Google API key
# Get your key from: https://makersuite.google.com/app/apikey
nano .env  # or use any text editor
```

Your `.env` file should look like:
```
GOOGLE_API_KEY=AIzaSy...your_actual_key_here
```

### 3. Prepare Your Data

Place your `creditcard.csv` file in the project directory.

```bash
# Build the vector database
python ingest_data.py
```

Expected output:
```
📁 Loading data from creditcard.csv...
✓ Loaded 284807 documents
✂️  Chunking data...
✓ Created 142403 chunks
🔨 Building Vector DB with Gemini embeddings...
✓ Database built successfully at ./chroma_db
```

### 4. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🎮 Using the Dashboard

### Basic Usage

1. **Enter a question** in the text area
2. **Select pipelines** to run (checkboxes in sidebar)
3. **Click "Run Selected Pipelines"** or **"Compare All"**
4. **View results** including:
   - Answer from each pipeline
   - Response time
   - Retrieved sources (for RAG pipelines)
   - Performance comparison

### Example Questions

- "What are the core metrics tracked in the dataset?"
- "Explain the main patterns in the data"
- "What anomalies or outliers exist?"
- "Summarize the key findings"

### Features

- ✅ Side-by-side pipeline comparison
- ✅ Response time tracking
- ✅ Source document inspection
- ✅ Example question templates
- ✅ Performance metrics
- ✅ Visual styling for easy differentiation

## 🔧 Testing Individual Pipelines

You can test each pipeline independently:

```bash
# Test Pipeline 1 (LLM-Only)
python pipeline_1_llm.py

# Test Pipeline 2 (Vector RAG)
python pipeline_2_rag.py

# Test Pipeline 3 (GraphRAG - placeholder)
python pipeline_3_graphrag.py
```

## 🌐 Pipeline 3: TigerGraph Integration

Pipeline 3 is currently a placeholder. To integrate TigerGraph GraphRAG:

### Setup Steps

1. **Install TigerGraph dependencies**:
```bash
pip install pyTigerGraph
```

2. **Add TigerGraph credentials to `.env`**:
```
TIGERGRAPH_HOST=your_host
TIGERGRAPH_GRAPH=your_graph_name
TIGERGRAPH_USERNAME=your_username
TIGERGRAPH_PASSWORD=your_password
```

3. **Implement the core functions** in `pipeline_3_graphrag.py`:
   - `extract_entities(question)` - Extract entities from question
   - `query_graph(entities)` - Query TigerGraph for subgraph
   - `format_graph_context(subgraph)` - Format for LLM
   - Replace placeholder code with actual TigerGraph calls

4. **Reference**: Check the TigerGraph GraphRAG repo for implementation details

## 📊 Hackathon Demo Tips

### Presentation Strategy

1. **Start with the problem**: Show limitations of LLM-only approach
2. **Demonstrate Pipeline 1**: Ask a question about your dataset - watch it fail
3. **Show Pipeline 2**: Same question - now it works with context
4. **Preview Pipeline 3**: Explain how graph relationships improve reasoning
5. **Compare metrics**: Show response times and quality differences

### Live Demo Flow

```
1. Open dashboard
2. Ask: "What patterns exist in fraudulent transactions?"
3. Run Pipeline 1 → Generic answer (no specific data)
4. Run Pipeline 2 → Specific answer with retrieved context
5. Show source documents from vector retrieval
6. Explain Pipeline 3 advantage → Multi-hop reasoning
```

### Key Talking Points

- **Pipeline 1**: Baseline - shows why retrieval is necessary
- **Pipeline 2**: Industry standard - fast and effective for most cases
- **Pipeline 3**: Advanced - handles complex queries requiring relationship reasoning

### Metrics to Highlight

- Response time comparison
- Context relevance (sources retrieved)
- Answer quality and specificity
- Scalability considerations

## 🛠️ Customization

### Change LLM Model

Edit `config.py`:
```python
LLM_MODEL = "gemini-1.5-pro"  # Use Pro instead of Flash for better quality
```

### Adjust Retrieval Settings

Edit `config.py`:
```python
RETRIEVAL_K = 8  # Retrieve more documents (default: 4)
CHUNK_SIZE = 1500  # Larger chunks (default: 1000)
```

### Use Different Data

Replace `creditcard.csv` with your own CSV and update in `config.py`:
```python
DATA_FILE = "your_data.csv"
```

Then rebuild the database:
```bash
python ingest_data.py
```

## 🐛 Troubleshooting

### API Key Issues
```
Error: GOOGLE_API_KEY not found
```
**Solution**: Make sure `.env` file exists and contains valid API key

### Database Not Found
```
Error: Database verification failed
```
**Solution**: Run `python ingest_data.py` to build the database

### Import Errors
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### ChromaDB Issues
```
Error: Collection not found
```
**Solution**: Delete `./chroma_db` folder and rebuild with `python ingest_data.py`

## 📈 Performance Optimization

### For Hackathon Demo

- Use `gemini-1.5-flash` (faster) for demos
- Pre-load database before presentation
- Test questions beforehand
- Have backup questions ready

### For Production

- Use `gemini-1.5-pro` for better quality
- Implement caching for repeated queries
- Add query preprocessing
- Implement streaming responses

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [Google AI Studio](https://makersuite.google.com/)
- [TigerGraph GraphRAG](https://github.com/TigerGraph-DevLabs/GraphRAG)

## 📝 License

This is a hackathon project template. Feel free to use and modify for your needs.

## 🤝 Contributing

This is designed as a hackathon starter. Customize it for your specific use case!

## ✨ Next Steps

1. ✅ Set up basic pipelines (Done!)
2. ✅ Create comparison dashboard (Done!)
3. 🔄 Integrate TigerGraph GraphRAG (Your task!)
4. 📊 Add evaluation metrics
5. 🎨 Enhance UI/UX
6. 🚀 Deploy for demo

---

**Good luck with your hackathon! 🚀**
