import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional

class FAISSStore:
    """FAISS vector store with metadata."""
    
    def __init__(self, dimension: int = 384, index_path: str = "./data/vector_store/index.faiss"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = index_path.replace(".faiss", "_meta.pkl")
        self.metadata: Dict[str, Any] = {}
        # Ordered list of chunk IDs that maps 1:1 with FAISS internal indices
        self._id_map: List[str] = []
        self._load_or_create()
    
    def _load_or_create(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                saved = pickle.load(f)
                self.metadata = saved.get("metadata", saved) if isinstance(saved, dict) and "metadata" in saved else saved
                self._id_map   = saved.get("id_map", list(self.metadata.keys())) if isinstance(saved, dict) and "id_map" in saved else list(self.metadata.keys())
        else:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine if normalized
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def add(self, chunks: List[Dict[str, Any]]):
        """Add chunks with embeddings."""
        embeddings = []
        for chunk in chunks:
            if "embedding" not in chunk:
                raise ValueError(f"Chunk {chunk['id']} missing embedding")
            chunk_id = chunk["id"]
            embeddings.append(chunk["embedding"])
            self.metadata[chunk_id] = chunk
            self._id_map.append(chunk_id)  # Track positional order
        
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)
        self._save()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            return []
        
        k = min(k, self.index.ntotal)
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self._id_map):
                continue
            chunk_id = self._id_map[idx]  # Use positional map — correct!
            if chunk_id not in self.metadata:
                continue
            chunk = self.metadata[chunk_id].copy()
            chunk["score"] = float(dist)
            results.append(chunk)
        
        return results
    
    def delete(self, chunk_ids: List[str]):
        """Remove chunks by ID. Requires rebuilding index."""
        id_set = set(chunk_ids)
        for cid in chunk_ids:
            if cid in self.metadata:
                del self.metadata[cid]
        self._id_map = [cid for cid in self._id_map if cid not in id_set]
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild FAISS index from metadata (preserving id_map order)."""
        self.index = faiss.IndexFlatIP(self.dimension)
        new_id_map = []
        if self._id_map:
            embeddings = []
            for cid in self._id_map:
                if cid in self.metadata:
                    embeddings.append(self.metadata[cid]["embedding"])
                    new_id_map.append(cid)
            self._id_map = new_id_map
            if embeddings:
                embeddings = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
        self._save()
    
    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({"metadata": self.metadata, "id_map": self._id_map}, f)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "total_chunks": len(self.metadata),
            "index_size": self.index.ntotal
        }