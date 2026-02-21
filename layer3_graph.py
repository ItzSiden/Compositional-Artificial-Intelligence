"""
LAYER 3: Concept Graph (NetworkX)
-----------------------------------
- Extracts nouns/keywords from user prompts
- Builds a weighted graph where co-mentioned words strengthen edges
- Retrieves top related concepts to inject into the prompt
"""

import os
import re
import json
import networkx as nx

GRAPH_FILE = "graph_data/concept_graph.json"

# Common English stopwords to filter out
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "how",
    "when", "where", "why", "all", "each", "every", "any", "some",
    "write", "create", "make", "show", "tell", "give", "get", "use",
    "help", "want", "need", "like", "also", "just", "more", "very"
}

class ConceptGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self._load_graph()
        print(f"[Layer 3] Concept graph loaded ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges).")

    def _load_graph(self):
        os.makedirs("graph_data", exist_ok=True)
        if os.path.exists(GRAPH_FILE):
            with open(GRAPH_FILE, "r") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)

    def _save_graph(self):
        data = nx.node_link_data(self.graph)
        with open(GRAPH_FILE, "w") as f:
            json.dump(data, f)

    def _extract_keywords(self, text: str) -> list:
        """Extract meaningful words from text (basic NLP without heavy deps)."""
        # Lowercase and extract words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_+#.-]{2,}\b', text.lower())
        # Filter stopwords and keep meaningful terms
        keywords = [w for w in words if w not in STOPWORDS]
        return list(set(keywords))  # Deduplicate

    def update(self, text: str):
        """Extract concepts from text and update the graph."""
        keywords = self._extract_keywords(text)

        # Add nodes
        for kw in keywords:
            if not self.graph.has_node(kw):
                self.graph.add_node(kw, mentions=1)
            else:
                self.graph.nodes[kw]["mentions"] = self.graph.nodes[kw].get("mentions", 0) + 1

        # Add/strengthen edges for co-mentioned keywords
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i+1:]:
                if kw1 != kw2:
                    if self.graph.has_edge(kw1, kw2):
                        self.graph[kw1][kw2]["weight"] += 1
                    else:
                        self.graph.add_edge(kw1, kw2, weight=1)

        self._save_graph()

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Given a query, find the top related concepts from the graph."""
        keywords = self._extract_keywords(query)
        related = {}

        for kw in keywords:
            if self.graph.has_node(kw):
                neighbors = self.graph[kw]
                for neighbor, attrs in neighbors.items():
                    weight = attrs.get("weight", 1)
                    related[neighbor] = related.get(neighbor, 0) + weight

        # Sort by total weight and return top_k
        sorted_concepts = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in sorted_concepts[:top_k]]

    def visualize(self):
        """Save a PNG visualization of the concept graph."""
        try:
            import matplotlib.pyplot as plt
            if self.graph.number_of_nodes() == 0:
                print("[Layer 3] Graph is empty — chat first to build it!")
                return

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.graph, k=2, seed=42)
            
            # Node sizes by mention count
            sizes = [self.graph.nodes[n].get("mentions", 1) * 200 for n in self.graph.nodes()]
            
            # Edge widths by weight
            weights = [self.graph[u][v].get("weight", 1) for u, v in self.graph.edges()]
            
            nx.draw_networkx_nodes(self.graph, pos, node_size=sizes, node_color="#4A90D9", alpha=0.8)
            nx.draw_networkx_labels(self.graph, pos, font_size=9, font_color="white", font_weight="bold")
            nx.draw_networkx_edges(self.graph, pos, width=weights, alpha=0.5, edge_color="#888")
            
            plt.title("MSCP Concept Graph", fontsize=16, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("concept_graph.png", dpi=120, bbox_inches="tight")
            plt.close()
            print("[Layer 3] Graph saved as concept_graph.png")
        except Exception as e:
            print(f"[Layer 3] Visualization error: {e}")
