"""
agent_pipeline.py
------------------
This is the CORE of the project: a multi-step pipeline that simulates
how an AI agent would approach a task, rather than just calling an LLM
once and returning whatever comes back.

----------------------------------------------------------------------
HOW THIS SIMULATES AN "AI AGENT WORKFLOW" (interview explanation)
----------------------------------------------------------------------
A true autonomous agent (e.g. using LangChain Agents or a ReAct loop)
decides FOR ITSELF which tools to call and in what order, often in a
loop, based on the LLM's own reasoning.

This project does NOT do that — and that's an honest, deliberate
design choice for clarity and reliability. Instead, it implements a
FIXED multi-step pipeline that mirrors the same underlying idea:

    Step 1: INPUT    -> capture and validate the user's request
    Step 2: RETRIEVE -> fetch relevant context (tool call: vector search)
    Step 3: REASON   -> build a prompt that combines context + task,
                         deciding HOW the LLM should use that context
    Step 4: GENERATE -> call the LLM (tool call: LLM API) to produce
                         the final output
    Step 5: LOG/RETURN -> record what happened at each step and return
                         a structured result

This is the same backbone that real agent frameworks automate — we're
just making each step EXPLICIT and deterministic instead of letting
the LLM choose the steps dynamically. This is genuinely how a lot of
production "agentic" pipelines start before teams introduce a full
agent framework: you get observability, predictability and lower cost,
at the expense of flexibility.

In an interview, this is the key sentence to say:
"I implemented an explicit retrieve-reason-generate pipeline rather
than a fully autonomous agent loop, because it's more predictable,
debuggable, and cheaper to run — and it's the same core pattern that
agent frameworks like LangChain abstract away."
----------------------------------------------------------------------
"""

from core.vector_store import VectorStore
from core.llm_client import LLMClient
from core.logger import get_logger

logger = get_logger(__name__)


class DocumentAgentPipeline:
    """
    Orchestrates the Retrieve -> Reason -> Generate workflow for three
    tasks: Q&A, Summarization, and Key Point extraction.

    One pipeline instance = one loaded document.
    """

    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm = llm_client

    # ------------------------------------------------------------------
    # STEP-BY-STEP LOGGED WRAPPER
    # ------------------------------------------------------------------
    def _log_step(self, step_name: str, detail: str = ""):
        logger.info(f"[AGENT STEP] {step_name} {('- ' + detail) if detail else ''}")

    # ------------------------------------------------------------------
    # TASK 1: Question Answering (classic RAG)
    # ------------------------------------------------------------------
    def answer_question(self, question: str) -> dict:
        """
        Full RAG flow for answering a question about the document.
        Returns a dict with the answer AND the intermediate steps,
        so the UI can show "what the agent did" transparently.
        """
        # STEP 1: INPUT
        self._log_step("INPUT", f"User question: {question}")
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # STEP 2: RETRIEVE
        self._log_step("RETRIEVE", "Searching vector store for relevant chunks")
        retrieved = self.vector_store.search(question)
        context_text = "\n\n---\n\n".join([r["chunk"] for r in retrieved])

        # STEP 3: REASON (prompt construction = where the "reasoning" plan happens)
        self._log_step("REASON", "Building grounded prompt from retrieved context")
        system_prompt = (
            "You are a precise document assistant. Answer the user's question "
            "using ONLY the provided context. If the answer isn't in the context, "
            "say so honestly instead of guessing."
        )
        user_prompt = (
            f"CONTEXT FROM DOCUMENT:\n{context_text}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer clearly and concisely based only on the context above."
        )

        # STEP 4: GENERATE
        self._log_step("GENERATE", "Calling LLM with grounded context")
        answer = self.llm.generate(system_prompt, user_prompt)

        self._log_step("DONE", "Returning answer to user")
        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "steps": ["INPUT", "RETRIEVE", "REASON", "GENERATE"],
        }

    # ------------------------------------------------------------------
    # TASK 2: Summarization
    # ------------------------------------------------------------------
    def summarize_document(self, all_chunks: list[str]) -> str:
        """
        Summarizes the full document.
        Note: for a large doc, we can't fit everything in one LLM call,
        so we use a simple "map-reduce" style approach:
          1. Summarize chunks in batches (map)
          2. Summarize the summaries (reduce)
        This keeps prompts within the model's context limit.
        """
        self._log_step("INPUT", "Summarize full document request")

        # STEP 2: RETRIEVE -> here "retrieval" is simply using all chunks,
        # since summarization needs the whole document, not a search result.
        self._log_step("RETRIEVE", f"Using all {len(all_chunks)} chunks")

        # MAP step: summarize in batches to respect context limits
        batch_size = 6
        batch_summaries = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_text = "\n\n".join(batch)

            self._log_step("REASON", f"Summarizing batch {i // batch_size + 1}")
            system_prompt = "You are a concise summarizer. Summarize the given text in 3-4 sentences."
            user_prompt = f"Text:\n{batch_text}\n\nSummary:"

            self._log_step("GENERATE", f"Calling LLM for batch {i // batch_size + 1}")
            batch_summary = self.llm.generate(system_prompt, user_prompt)
            batch_summaries.append(batch_summary)

        # REDUCE step: combine batch summaries into one final summary
        combined = "\n\n".join(batch_summaries)
        self._log_step("REASON", "Combining batch summaries into final summary")
        system_prompt = "You are a concise summarizer. Combine these partial summaries into one coherent overall summary (5-8 sentences)."
        user_prompt = f"Partial summaries:\n{combined}\n\nFinal combined summary:"

        self._log_step("GENERATE", "Calling LLM for final summary")
        final_summary = self.llm.generate(system_prompt, user_prompt)

        self._log_step("DONE", "Returning final summary")
        return final_summary

    # ------------------------------------------------------------------
    # TASK 3: Key Point Extraction
    # ------------------------------------------------------------------
    def extract_key_points(self, all_chunks: list[str], num_points: int = 6) -> str:
        """
        Extracts key points/bullets from the document.
        Reuses the same map-reduce style as summarization, but asks
        for bullet points instead of prose.
        """
        self._log_step("INPUT", "Extract key points request")
        self._log_step("RETRIEVE", f"Using all {len(all_chunks)} chunks")

        # For a portfolio-scale doc we can usually fit a sample of chunks
        # directly. For very large docs, the same map-reduce idea from
        # summarize_document() applies.
        sample_text = "\n\n".join(all_chunks[:15])  # cap to stay within limits

        self._log_step("REASON", "Building key-point extraction prompt")
        system_prompt = (
            "You are an expert at extracting key insights from documents. "
            "Read the text and extract the most important points as a clean bullet list."
        )
        user_prompt = (
            f"Text:\n{sample_text}\n\n"
            f"Extract the top {num_points} key points as a bullet list. "
            f"Each bullet should be one clear, concise sentence."
        )

        self._log_step("GENERATE", "Calling LLM for key points")
        key_points = self.llm.generate(system_prompt, user_prompt)

        self._log_step("DONE", "Returning key points")
        return key_points
