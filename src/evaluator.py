"""
evaluator.py
------------
Reusable evaluation framework for LLM behavior analysis.

Classes:
    LLMEvaluator: Hallucination, consistency, and response scoring.
"""

import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LLMEvaluator:
    """Evaluate LLM behavior across multiple dimensions."""

    def __init__(self, model: str = "gemma3:12b",
                 ollama_url: str = "http://127.0.0.1:11434",
                 embed_model: str = "all-MiniLM-L6-v2"):
        self.model = model
        self.ollama_url = ollama_url
        self.embed_model = SentenceTransformer(embed_model)

    def query(self, prompt: str) -> str:
        """Send a prompt to the local Ollama model."""
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            proxies={"http": None, "https": None}
        )
        return response.json()["message"]["content"]

    def semantic_score(self, response: str, reference: str) -> float:
        """
        Score response quality by semantic similarity to reference.

        Args:
            response: Model generated answer
            reference: Ground truth answer

        Returns:
            Cosine similarity score between 0 and 1
        """
        r_emb = self.embed_model.encode([response])
        ref_emb = self.embed_model.encode([reference])
        return float(cosine_similarity(r_emb, ref_emb)[0][0])

    def hallucination_check(self, question: str, context: str) -> dict:
        """
        Check if model hallucinates when answer is not in context.

        Args:
            question: Question to ask
            context: Context that does NOT contain the answer

        Returns:
            Dict with response and hallucination flag
        """
        prompt = f"""Answer the question based ONLY on the context below.
If the answer is not in the context, say exactly: "I don't know based on the provided context."

Context:
{context}

Question: {question}
Answer:"""

        response = self.query(prompt)
        refused = any(phrase in response.lower() for phrase in [
            "i don't know",
            "not in the context",
            "cannot find",
            "not mentioned",
            "no information"
        ])

        return {
            "question": question,
            "response": response,
            "refused": refused,
            "hallucinated": not refused
        }

    def consistency_score(self, question_variants: list, context: str) -> dict:
        """
        Measure answer consistency across question variants.

        Args:
            question_variants: List of semantically identical questions
            context: Context to answer from

        Returns:
            Dict with answers, similarity matrix, and avg consistency score
        """
        answers = []
        for q in question_variants:
            prompt = f"Context: {context}\nQuestion: {q}\nAnswer:"
            answers.append(self.query(prompt))

        embeddings = self.embed_model.encode(answers)
        sim_matrix = cosine_similarity(embeddings)
        n = len(answers)
        avg_sim = (sim_matrix.sum() - n) / (n ** 2 - n)

        return {
            "answers": answers,
            "similarity_matrix": sim_matrix,
            "consistency_score": float(avg_sim)
        }