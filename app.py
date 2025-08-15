import os
import fitz
import faiss
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer, util

# Load embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Store FAISS index and chunks in memory
current_chunks = []
current_index = None

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def semantic_chunking(text, threshold=0.7):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    embeddings = embedding_model.encode(paragraphs)
    chunks, current_chunk = [], [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        similarity = util.cos_sim(embeddings[i], embeddings[i-1]).item()
        if similarity >= threshold:
            current_chunk.append(paragraphs[i])
        else:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraphs[i]]
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks

def create_faiss_index(chunks):
    vectors = embedding_model.encode(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    return index

def get_top_k_chunks(query, chunks, index, k=3):
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), k)
    return [chunks[i] for i in I[0]]

def ask_llm_question(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
Use the context below to answer the question in a professional manner.

Context:
{context}

Question: {question}
Answer:"""

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

@app.route("/", methods=["GET", "POST"])
def index():
    global current_chunks, current_index

    if request.method == "POST":
        if "pdf" in request.files:
            pdf_file = request.files["pdf"]
            if pdf_file.filename != "":
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
                pdf_file.save(pdf_path)

                # Process PDF
                text = extract_text_from_pdf(pdf_path)
                current_chunks = semantic_chunking(text)
                current_index = create_faiss_index(current_chunks)

                return render_template("index.html", message="PDF uploaded and processed successfully!")

        if "question" in request.form:
            question = request.form["question"]
            if question.strip() != "" and current_index is not None:
                top_chunks = get_top_k_chunks(question, current_chunks, current_index)
                answer = ask_llm_question(question, top_chunks)
                return render_template("index.html", answer=answer, question=question)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
