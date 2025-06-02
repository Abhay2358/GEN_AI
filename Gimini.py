import fitz  # PyMuPDF
import google.generativeai as genai # Import the Google Generative AI library
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os # Import os to access environment variables

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text


def semantic_chunking(text, threshold=0.7):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    embeddings = embedding_model.encode(paragraphs)
    
    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        similarity = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
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
    return index, vectors


def get_top_k_chunks(query, chunks, index, vectors, k=3):
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), k)
    return [chunks[i] for i in I[0]]


def ask_llm_question(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant. Use only the information from the context below to answer the user's question.

Context:
{context}

Question: {question}
Answer:"""

    # Configure the Gemini API key
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash') 

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=500
        )
    )
    return response.text


# 🔽 Example usage
if __name__ == "__main__":
    pdf_path = "/content/5-mb-example-file.pdf" 

    print("[1] Extracting PDF text...")
    text = extract_text_from_pdf(pdf_path)

    print("[2] Chunking text semantically...")
    chunks = semantic_chunking(text)

    print(f"[3] Creating vector index from {len(chunks)} chunks...")
    index, vectors = create_faiss_index(chunks)

    # --- Start of the while loop for continuous questioning ---
    while True:
        user_question = input("\nAsk your question (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Exiting RAG system. Goodbye!")
            break

        print("[4] Retrieving top chunks related to your question...")
        top_chunks = get_top_k_chunks(user_question, chunks, index, vectors, k=3)

        print("[5] Asking LLM...")
        answer = ask_llm_question(user_question, top_chunks)

        print("\n[✅ Answer]")
        print(answer)
    # --- End of the while loop ---