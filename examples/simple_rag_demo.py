"""
A small, complete RAG (Retrieval-Augmented Generation) system, built from
scratch, with hallufirewall running on top of it to catch hallucinations.

This demonstrates the FULL real-world use case: a knowledge base, a
retriever that finds relevant passages for a question, an LLM that
generates an answer, and the firewall checking that answer before it
would be shown to a user.

NEW CONCEPT: embeddings + cosine similarity (used for retrieval)
    An "embedding" is a list of numbers (a vector) that represents the
    MEANING of a piece of text -- similar meanings produce similar vectors.
    To find relevant documents for a question, we:
      1. Convert the question into an embedding
      2. Convert every document into an embedding (done once, upfront)
      3. Measure how "close" the question's vector is to each document's
         vector (using cosine similarity -- closer to 1.0 = more similar)
      4. Return the top-K closest documents as the retrieved context
    This is exactly what real vector databases (Pinecone, Chroma, FAISS)
    do under the hood, just at a much larger scale.
"""

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np

from hallufirewall import Firewall

load_dotenv()
client = genai.Client()


def gemini_llm_client(prompt: str) -> str:
    response = client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
    return response.text


# ---------------------------------------------------------------------------
# 1. THE KNOWLEDGE BASE
# A small set of documents our RAG system can retrieve from. Each one is
# a short, self-contained fact -- deliberately spanning a few different
# topics so retrieval has to actually pick the RIGHT ones for a question.
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "Paris is the capital of France. It is located on the Seine river.",
    "Paris has a population of just over 2 million people within the city limits.",
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    "Aspirin is not recommended for children under 16 due to the risk of Reye's syndrome.",
    "Ibuprofen should be taken with food to reduce the risk of stomach irritation.",
    "The standard adult dose of paracetamol is 500-1000mg every 4-6 hours, not exceeding 4000mg per day.",
    "Python is a programming language created by Guido van Rossum, first released in 1991.",
    "The Python Global Interpreter Lock (GIL) prevents multiple native threads from executing Python bytecode simultaneously.",
    "Mount Everest is the tallest mountain above sea level, at 8,849 meters.",
    "The Great Barrier Reef is located off the coast of Queensland, Australia.",
]


# ---------------------------------------------------------------------------
# 2. THE RETRIEVER
# ---------------------------------------------------------------------------
class SimpleRetriever:
    def __init__(self, documents: list[str]):
        print("Loading embedding model... (only happens once)")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents
        # Compute embeddings for every document ONCE, upfront, and cache them.
        # This is what a real vector database does at "indexing time."
        self.doc_embeddings = self.model.encode(documents)
        print(f"Indexed {len(documents)} documents.\n")

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_embedding = self.model.encode([query])[0]

        # Cosine similarity: measures the ANGLE between two vectors,
        # ignoring their length. 1.0 = identical direction (very similar
        # meaning), 0.0 = unrelated, -1.0 = opposite meaning.
        similarities = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get the indices of the top-K most similar documents.
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]


# ---------------------------------------------------------------------------
# 3. THE GENERATOR (answers using ONLY the retrieved context)
# ---------------------------------------------------------------------------
RAG_PROMPT = """Answer the question using the CONTEXT below. Be helpful and complete.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def generate_answer(question: str, context_chunks: list[str]) -> str:
    prompt = RAG_PROMPT.format(context="\n".join(context_chunks), question=question)
    return gemini_llm_client(prompt)


# ---------------------------------------------------------------------------
# 4. PUTTING IT ALL TOGETHER: retrieve -> generate -> firewall check
# ---------------------------------------------------------------------------
def ask(question: str, retriever: SimpleRetriever, firewall: Firewall):
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 70}")

    context = retriever.retrieve(question, top_k=3)
    print("\nRetrieved context:")
    for c in context:
        print(f"  - {c}")

    answer = generate_answer(question, context)
    print(f"\nGenerated answer:\n  {answer}")

    result = firewall.check(answer=answer, context=context)
    print(f"\nFIREWALL VERDICT: {result.verdict.upper()}  (confidence: {result.overall_confidence:.2f})")
    for cv in result.claim_verdicts:
        print(f"  - \"{cv.claim.text}\" -> {cv.label.value} ({cv.confidence:.2f})")

    return result


if __name__ == "__main__":
    retriever = SimpleRetriever(KNOWLEDGE_BASE)
    firewall = Firewall(llm_client=gemini_llm_client, claim_extraction_mode="llm")

    # A few pre-set test questions.
    ask("What is the capital of France and what river is it on?", retriever, firewall)
    ask("What is the maximum daily dose of paracetamol for an adult?", retriever, firewall)
    ask("How tall is Mount Everest?", retriever, firewall)

    # Interactive mode: ask your own questions.
    print(f"\n{'=' * 70}")
    print("Interactive mode -- type your own question, or 'exit' to quit.")
    print(f"{'=' * 70}")
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ("exit", "quit"):
            break
        if user_question:
            ask(user_question, retriever, firewall)
