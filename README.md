# RAG System - PDF Question Answering with Semantic Search

A Retrieval-Augmented Generation (RAG) system built with Flask that enables intelligent question-answering from PDF documents using semantic chunking, FAISS vector search, and Google's Gemini AI.

## 🌟 Features

- **PDF Document Processing**: Upload and extract text from PDF files using PyMuPDF
- **Semantic Chunking**: Intelligent text splitting based on semantic similarity between paragraphs
- **FAISS Vector Search**: Fast and efficient similarity search using Facebook's FAISS library
- **Context-Aware Responses**: Generate accurate answers using Google's Gemini 2.0 Flash model
- **Web Interface**: Simple and intuitive Flask-based UI for document upload and querying
- **In-Memory Processing**: Fast retrieval with in-memory FAISS indexing

## 🏗️ System Architecture

```
PDF Upload → Text Extraction → Semantic Chunking → Embedding Generation → FAISS Index
                                                                              ↓
User Query → Query Embedding → Similarity Search → Top-K Retrieval → Gemini AI → Answer
```

## 🛠️ Technologies Used

- **Backend Framework**: Flask 3.1.1
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: PyMuPDF (fitz)
- **ML Libraries**: 
  - sentence-transformers 4.1.0
  - torch 2.7.0
  - numpy 2.2.6
  - scikit-learn 1.6.1

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API Key

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Abhay2358/GEN_AI.git
cd GEN_AI
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirement.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory or set the environment variable:
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

Or on Windows:
```cmd
set GEMINI_API_KEY=your_gemini_api_key_here
```

## 💻 Usage

### Running the Application

1. **Start the Flask server**
```bash
python app.py
```

2. **Access the web interface**
   - Open your browser and navigate to `http://127.0.0.1:5000`

3. **Upload a PDF**
   - Click the upload button and select a PDF file
   - Wait for the processing confirmation message

4. **Ask Questions**
   - Enter your question in the text field
   - Click submit to get AI-generated answers based on the PDF content

### How It Works

1. **Document Processing**: The system extracts text from uploaded PDFs and splits it into semantically coherent chunks using cosine similarity (threshold: 0.7)

2. **Indexing**: Each chunk is converted to embeddings using the `all-MiniLM-L6-v2` model and stored in a FAISS index

3. **Query Processing**: When you ask a question:
   - Your question is converted to an embedding
   - FAISS searches for the top 3 most relevant chunks
   - These chunks are sent as context to Gemini AI
   - Gemini generates a comprehensive answer

## 📂 Project Structure

```
GEN_AI/
├── app.py                 # Main Flask application
├── requirement.txt        # Python dependencies
├── uploads/              # Directory for uploaded PDFs
├── templates/
│   └── index.html        # Web interface template
└── README.md             # Project documentation
```

## 🔧 Configuration

You can customize the following parameters in `app.py`:

```python
# Semantic chunking threshold (0-1)
threshold = 0.7  # Higher = more similar paragraphs grouped together

# Number of chunks to retrieve for context
k = 3  # Retrieve top 3 most relevant chunks

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')
```

## 🎯 Key Functions

### `extract_text_from_pdf(pdf_path)`
Extracts all text content from a PDF file using PyMuPDF.

### `semantic_chunking(text, threshold=0.7)`
Splits text into semantically coherent chunks based on paragraph similarity.

### `create_faiss_index(chunks)`
Creates a FAISS L2 index for efficient similarity search.

### `get_top_k_chunks(query, chunks, index, k=3)`
Retrieves the k most relevant chunks for a given query.

### `ask_llm_question(question, context_chunks)`
Generates an answer using Gemini AI with retrieved context.

## 📊 Performance

- **Embedding Model Dimension**: 384 (all-MiniLM-L6-v2)
- **Search Algorithm**: FAISS IndexFlatL2 (exact L2 distance)
- **Default Retrieval**: Top 3 most relevant chunks
- **Supported Format**: PDF documents
- **Processing Speed**: Depends on PDF size and complexity

## 🚀 Deployment

### Local Deployment
```bash
python app.py
```

### Production Deployment (using Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 🧪 Example Use Cases

- **Academic Research**: Query research papers and get quick answers
- **Legal Documents**: Extract information from contracts and legal texts
- **Technical Manuals**: Find specific instructions in user manuals
- **Business Reports**: Analyze and query business documents
- **Educational Materials**: Study from textbooks and course materials

## 🔒 Security Notes

- Store your `GEMINI_API_KEY` securely using environment variables
- Never commit API keys to version control
- Consider adding file size limits for PDF uploads
- Implement authentication for production use

## 📈 Future Enhancements

- [ ] Support for multiple document types (DOCX, TXT, etc.)
- [ ] Persistent vector storage (ChromaDB, Pinecone)
- [ ] Multi-document querying
- [ ] Conversation history and follow-up questions
- [ ] Advanced chunking strategies (overlap, fixed-size)
- [ ] User authentication and session management
- [ ] Citation tracking (show which chunk answered the question)
- [ ] Batch question processing
- [ ] Export answers to PDF/TXT
- [ ] Support for other LLMs (OpenAI, Anthropic Claude)

## 🐛 Troubleshooting

**Issue**: "GEMINI_API_KEY not found"
- **Solution**: Ensure you've set the environment variable correctly

**Issue**: PDF not processing
- **Solution**: Check if the PDF is readable and not password-protected

**Issue**: Out of memory errors
- **Solution**: Process smaller PDFs or increase system memory

**Issue**: Slow response times
- **Solution**: Reduce the number of chunks retrieved (k parameter) or use a smaller embedding model

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini**: For the powerful LLM API
- **Sentence Transformers**: For the efficient embedding models
- **FAISS**: For lightning-fast similarity search
- **PyMuPDF**: For reliable PDF text extraction
- **Flask**: For the simple and elegant web framework

## 📧 Contact

**Abhay** - [@Abhay2358](https://github.com/Abhay2358)

Project Link: [https://github.com/Abhay2358/GEN_AI](https://github.com/Abhay2358/GEN_AI)

## 📖 Additional Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
