"""
LAYER 2: Vector Brain (FAISS + MiniLM)
---------------------------------------
- Embeds text using all-MiniLM-L6-v2 (only ~80MB)
- Stores embeddings in a FAISS index (brain.index)
- Retrieves the 3 most relevant text chunks for any prompt
- Also used by Layer 4 to store compressed long-term memories
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BRAIN_INDEX_FILE = "brain.index"
BRAIN_META_FILE  = "brain_meta.pkl"
KNOWLEDGE_DIR    = "knowledge_base"
MODEL_NAME       = "all-MiniLM-L6-v2"
EMBEDDING_DIM    = 384

class VectorBrain:
    def __init__(self):
        print("[Layer 2] Loading sentence embedding model...")
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.metadata = []   # Stores the original text for each vector
        self._load_or_create_index()
        self._ingest_knowledge_base()

    def _load_or_create_index(self):
        if os.path.exists(BRAIN_INDEX_FILE) and os.path.exists(BRAIN_META_FILE):
            self.index = faiss.read_index(BRAIN_INDEX_FILE)
            with open(BRAIN_META_FILE, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[Layer 2] Loaded existing brain ({len(self.metadata)} chunks).")
        else:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.metadata = []
            print("[Layer 2] Created new FAISS brain index.")

    def _save_index(self):
        faiss.write_index(self.index, BRAIN_INDEX_FILE)
        with open(BRAIN_META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def _ingest_knowledge_base(self):
        """Load all .txt files from knowledge_base/ folder into FAISS."""
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
            return

        new_chunks = 0
        for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(KNOWLEDGE_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # Split into ~200 word chunks with overlap
            chunks = self._chunk_text(text)
            for chunk in chunks:
                if chunk not in self.metadata:  # Avoid duplicates
                    self.add(chunk)
                    new_chunks += 1

        if new_chunks > 0:
            print(f"[Layer 2] Ingested {new_chunks} new chunks from knowledge_base/.")
            self._save_index()
        else:
            print(f"[Layer 2] Knowledge base up to date ({len(self.metadata)} chunks total).")

    def _chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 30) -> list:
        """Split text into overlapping word chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def add(self, text: str):
        """Add a text string to the FAISS index."""
        embedding = self.encoder.encode([text], convert_to_numpy=True).astype("float32")
        self.index = self.index if self.index else faiss.IndexFlatL2(EMBEDDING_DIM)
        self.index.add(embedding)
        self.metadata.append(text)

    def add_and_save(self, text: str):
        """Add text and persist to disk (used by Layer 4 for memory compression)."""
        self.add(text)
        self._save_index()

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Return the top_k most relevant text chunks for the query."""
        if self.index.ntotal == 0:
            return []
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results
