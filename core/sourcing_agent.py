"""
sourcing_agent.py
------------------
Think9 PoC addition.

This is deliberately a THIN agent: it reuses the exact same VectorStore
and LLMClient as the Institutional Memory Agent (DocumentAgentPipeline) —
no new retrieval logic, no new embedding model, no new index.

That's the point. It exists to prove the architecture's central claim:
once the shared retrieval + memory core is built, a new specialized agent
is just a new system prompt + a different question framing on top of the
SAME core. Adding agent #3, #4, #5 at Think9 (Consumer Insight, Feedback
Triage, ...) follows this identical pattern.

A production version of this agent would add real domain logic (e.g.
structured MOQ/lead-time extraction, cross-brand vendor deduplication) —
that's explicitly out of scope for this PoC and called out as "designed,
not built" in the architecture doc.
"""

from core.vector_store import VectorStore
from core.llm_client import LLMClient
from core.logger import get_logger
from core.config import RELEVANCE_DISTANCE_THRESHOLD

logger = get_logger(__name__)


class SourcingAgentPipeline:
    """
    Same shared retrieval core as DocumentAgentPipeline, different job:
    flag vendor/sourcing risks and bundling opportunities instead of
    answering a general question.
    """

    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm = llm_client

    def _log_step(self, step_name: str, detail: str = ""):
        logger.info(f"[SOURCING AGENT STEP] {step_name} {('- ' + detail) if detail else ''}")

    def analyze_sourcing_query(self, query: str, metadata_filter: dict | None = None) -> dict:
        """
        Example query: "Are there any vendors we could bundle volume across
        brands with?" or "What's our MOQ exposure with Vendor X?"
        """
        self._log_step("INPUT", f"Sourcing query: {query}")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        self._log_step("RETRIEVE", "Searching shared memory for vendor/sourcing context")
        # NOTE: same vector_store.search() call as the Memory Agent — this
        # line is the entire proof of "pluggable on the same core".
        retrieved = self.vector_store.search(query, top_k=5, metadata_filter=metadata_filter)

        best_distance = min((r["distance"] for r in retrieved), default=float("inf"))
        if not retrieved or best_distance > RELEVANCE_DISTANCE_THRESHOLD:
            self._log_step("DONE", "Insufficient sourcing context — flagged for human review")
            return {
                "answer": ("Not enough vendor/sourcing context in the shared memory to answer "
                           "this confidently. Flagged for the sourcing team to review."),
                "retrieved_chunks": retrieved,
                "confidence": "low",
                "needs_human_review": True,
                "agent": "sourcing_vendor_agent",
            }

        context_text = "\n\n---\n\n".join([r["chunk"] for r in retrieved])

        # DIFFERENT system prompt from the Memory Agent — same retrieval, different lens.
        system_prompt = (
            "You are a sourcing and vendor-risk analyst for a multi-brand consumer "
            "goods company. Given context pulled from vendor notes, quotes, and "
            "meeting records across multiple brands, identify: (1) any risks "
            "(supply delays, single-vendor dependency, unfavorable terms), and "
            "(2) any volume-bundling or renegotiation opportunities across brands. "
            "Be specific and cite which brand each point relates to. If the context "
            "doesn't support a finding, say so honestly."
        )
        user_prompt = (
            f"CONTEXT FROM SHARED MEMORY (vendor/sourcing documents across brands):\n{context_text}\n\n"
            f"SOURCING QUESTION: {query}\n\n"
            f"Give a structured answer: Risks found, then Opportunities found."
        )

        self._log_step("GENERATE", "Calling LLM with sourcing-analyst framing")
        answer = self.llm.generate(system_prompt, user_prompt)

        self._log_step("DONE", "Returning sourcing analysis")
        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "confidence": "high",
            "needs_human_review": False,
            "agent": "sourcing_vendor_agent",
        }
