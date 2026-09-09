"""
FAISS Vector Index Builder.
Creates embeddings and FAISS index for historical case retrieval.
Run: python -m src.retrieval.index
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Use a lightweight model for fast embedding
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def create_embedding_documents(cases_df):
    """
    Create embedding documents from cases.
    
    Combines customer problem + resolution for each case into a single
    document for embedding.
    
    Args:
        cases_df: DataFrame of cases
        
    Returns:
        List of embedding document strings and metadata
    """
    documents = []
    metadata = []
    
    for idx, row in cases_df.iterrows():
        # Create document text
        customer_problem = str(row.get('customer_problem', ''))
        resolution = str(row.get('resolution', ''))
        
        # Combine for embedding
        doc_text = f"Customer problem: {customer_problem}\nHistorical resolution: {resolution}"
        documents.append(doc_text)
        
        # Metadata
        meta = {
            'case_id': row.get('case_id', f'C{idx:06d}'),
            'conversation_id': row.get('conversation_id', ''),
            'intent': row.get('intent', 'general_inquiry'),
            'resolution_quality': row.get('resolution_quality', 'weak'),
            'turn_count': row.get('turn_count', 2),
        }
        metadata.append(meta)
    
    return documents, metadata


def build_faiss_index(cases_path, output_dir, sample_size=None):
    """
    Build FAISS index from cases.
    
    Args:
        cases_path: Path to cases CSV
        output_dir: Directory to save index and metadata
        sample_size: If provided, sample this many cases
    """
    print(f"Loading cases from {cases_path}...")
    cases = pd.read_csv(cases_path, low_memory=False)
    
    if sample_size and sample_size < len(cases):
        print(f"Sampling {sample_size:,} cases...")
        cases = cases.sample(n=sample_size, random_state=42)
    
    print(f"Building index for {len(cases):,} cases...")
    
    # Create embedding documents
    print("Creating embedding documents...")
    documents, metadata = create_embedding_documents(cases)
    
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"Index size: {index.ntotal:,} vectors")
    print(f"Embedding dimension: {dimension}")
    
    # Save index and metadata
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    metadata_path = os.path.join(output_dir, "index_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")
    
    # Save documents for reference
    docs_path = os.path.join(output_dir, "embedding_documents.json")
    with open(docs_path, 'w', encoding='utf-8') as f:
        json.dump(documents[:100], f, indent=2, ensure_ascii=False)  # Save first 100 for reference
    print(f"Saved sample documents to {docs_path}")
    
    # Save the case data as well
    cases_sample_path = os.path.join(output_dir, "cases_for_index.csv")
    cases.to_csv(cases_sample_path, index=False)
    print(f"Saved cases to {cases_sample_path}")
    
    return index, metadata, model


def main():
    """Build the FAISS index."""
    cases_path = "data/kb/amazonhelp_cases.csv"
    output_dir = "artifacts/vector_index"
    
    # Use a sample for faster builds (increase for production)
    build_faiss_index(cases_path, output_dir, sample_size=10000)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
