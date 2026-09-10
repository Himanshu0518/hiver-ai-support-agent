"""
Retrieval Module.
Searches FAISS index for similar historical cases.
"""
import logging
import numpy as np
import os
import sys
import json
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

log = logging.getLogger(__name__)


class Retriever:
    """Retrieves similar historical cases using FAISS."""

    def __init__(self, index_dir="artifacts/vector_index", model_name='all-MiniLM-L6-v2'):
        """
        Initialize the retriever.

        Args:
            index_dir: Directory containing FAISS index and metadata
            model_name: Sentence-transformer model name
        """
        self.index_dir = index_dir
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = None
        self.cases_df = None

    def load(self):
        """Load index, metadata, and model."""
        log.info("Loading retriever from %s...", self.index_dir)

        # Load embedding model
        log.info("Loading embedding model: %s...", self.model_name)
        self.model = SentenceTransformer(self.model_name)

        # Load FAISS index
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        log.info("Loading FAISS index from %s...", index_path)
        self.index = faiss.read_index(index_path)
        log.info("Index size: %s vectors", f"{self.index.ntotal:,}")

        # Load metadata
        metadata_path = os.path.join(self.index_dir, "index_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Load cases
        cases_path = os.path.join(self.index_dir, "cases_for_index.csv")
        self.cases_df = __import__('pandas').read_csv(cases_path, low_memory=False)

        log.info("Retriever loaded successfully.")
    
    def retrieve(self, query, top_k=5, intent_filter=None):
        """
        Retrieve similar cases for a query.
        
        Args:
            query: Customer message or query text
            top_k: Number of results to return
            intent_filter: Optional intent to filter results
            
        Returns:
            List of dicts with case info and similarity scores
        """
        if self.model is None:
            self.load()
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        # Search more than needed if filtering
        search_k = min(top_k * 3, self.index.ntotal) if intent_filter else top_k
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            meta = self.metadata[idx]
            case = self.cases_df.iloc[idx]
            
            # Apply intent filter if specified
            if intent_filter and meta.get('intent') != intent_filter:
                continue
            
            result = {
                'case_id': meta.get('case_id', ''),
                'conversation_id': meta.get('conversation_id', ''),
                'intent': meta.get('intent', ''),
                'resolution_quality': meta.get('resolution_quality', ''),
                'turn_count': meta.get('turn_count', 0),
                'similarity_score': float(score),
                'customer_problem': str(case.get('customer_problem', '')),
                'resolution': str(case.get('resolution', '')),
                'amazon_responses': str(case.get('amazon_responses', '')),
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_tf_idf(self, query, top_k=5, intent_filter=None):
        """
        Fallback TF-IDF retrieval when FAISS is not available.
        
        Args:
            query: Customer message
            top_k: Number of results
            intent_filter: Optional intent filter
            
        Returns:
            List of case dicts
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if self.cases_df is None:
            cases_path = os.path.join(self.index_dir, "cases_for_index.csv")
            self.cases_df = __import__('pandas').read_csv(cases_path, low_memory=False)
        
        df = self.cases_df
        if intent_filter:
            df = df[df['intent'] == intent_filter]
        
        if len(df) == 0:
            return []
        
        # Create TF-IDF
        texts = df['customer_problem'].fillna('').astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
            query_vec = tfidf_matrix[-1]
            doc_vecs = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] < 0.01:
                    continue
                row = df.iloc[idx]
                results.append({
                    'case_id': row.get('case_id', ''),
                    'conversation_id': row.get('conversation_id', ''),
                    'intent': row.get('intent', ''),
                    'resolution_quality': row.get('resolution_quality', ''),
                    'turn_count': row.get('turn_count', 0),
                    'similarity_score': float(similarities[idx]),
                    'customer_problem': str(row.get('customer_problem', '')),
                    'resolution': str(row.get('resolution', '')),
                    'amazon_responses': str(row.get('amazon_responses', '')),
                })
            return results
        except Exception as e:
            log.warning("TF-IDF retrieval error: %s", e)
            return []
