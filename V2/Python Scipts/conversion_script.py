import json
import os
import uuid
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# --------------------------------------------
# STEP 1: Load KMBC Crawl Data
# --------------------------------------------
def load_kmbc_data(filepath):
    print(f"📥 Loading KMBC crawl data from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --------------------------------------------
# STEP 2: Chunking Function
# --------------------------------------------
def chunk_text(text, max_tokens=500):
    """
    Breaks a large body of text into chunks, based on sentences,
    with each chunk roughly below a max token count.
    """
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
        current_chunk.append(sentence)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# --------------------------------------------
# STEP 3: Embedding Function
# --------------------------------------------
def embed_chunks(model, texts):
    print(f"🔢 Embedding {len(texts)} chunk(s)...")
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# --------------------------------------------
# STEP 4: Save FAISS Index + Metadata
# --------------------------------------------
def save_faiss_index(vectors, metadata_list, index_path="faiss_index"):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save index
    faiss.write_index(index, f"{index_path}.index")
    print(f"💾 FAISS index saved to {index_path}.index")

    # Save metadata
    with open(f"{index_path}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"📝 Metadata saved to {index_path}_metadata.json")

# --------------------------------------------
# MAIN PIPELINE
# --------------------------------------------
def process_kmbc_pipeline(json_path, faiss_path="faiss_index"):
    data = load_kmbc_data(json_path)
    print("🤖 Loading sentence embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # You can switch to a local one later

    all_embeddings = []
    all_metadata = []
    total_chunks = 0

    print(f"📄 Processing {len(data)} pages...\n")

    for entry in tqdm(data, desc="📚 Pages processed"):
        text_content = "\n\n".join([
            entry.get("summary", ""),
            entry.get("intro", ""),
            entry.get("text", ""),
            entry.get("alert", "")
        ]).strip()

        if not text_content:
            print(f"⚠️ Skipping empty page: {entry.get('url')}")
            continue

        chunks = chunk_text(text_content)
        if not chunks:
            print(f"⚠️ No chunks generated from: {entry.get('url')}")
            continue

        embeddings = embed_chunks(model, chunks)
        all_embeddings.append(embeddings)

        for i, chunk in enumerate(chunks):
            all_metadata.append({
                "id": str(uuid.uuid4()),
                "url": entry.get("url"),
                "title": entry.get("title"),
                "breadcrumb": " > ".join(entry.get("topics", [])),
                "department": entry.get("department"),
                "keywords": entry.get("keywords", []),
                "emails": entry.get("email", []),
                "phones": entry.get("phone", []),
                "links": entry.get("links", []),
                "text": chunk
            })
            total_chunks += 1

    if total_chunks == 0:
        print("❌ No content was processed. Exiting.")
        return

    final_vectors = np.vstack(all_embeddings)
    print(f"\n🧠 Total chunks processed: {total_chunks}")
    save_faiss_index(final_vectors, all_metadata, faiss_path)
    print("✅ All done!")

# --------------------------------------------
# ENTRY POINT
# --------------------------------------------
if __name__ == "__main__":
    import nltk
    print("⬇️ Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # File paths
    INPUT_JSON = "KMBC_crawl_data.json"
    OUTPUT_INDEX = "faiss_index"

    print("🚀 Starting KMBC vector index pipeline...\n")
    process_kmbc_pipeline(INPUT_JSON, OUTPUT_INDEX)
