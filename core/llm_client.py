"""
llm_client.py
-------------
Thin wrapper around the Groq API (OpenAI-compatible chat completions).

Why Groq instead of OpenAI here:
Groq offers a generous FREE tier and hosts open-source models
(like Llama 3.1) at very high speed. The API shape is identical to
OpenAI's `chat.completions.create(...)`, so this code would need
almost NO changes to swap to OpenAI/Azure OpenAI later — which is a
good thing to point out in an interview: "the integration is
provider-agnostic by design."
"""

from groq import Groq
from core.config import GROQ_API_KEY, GROQ_MODEL
from core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Create a .env file with "
                "GROQ_API_KEY=your_key_here (get a free key at console.groq.com)"
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """
        Sends a system + user prompt to the LLM and returns the text response.

        - system_prompt: sets the LLM's role/behavior (e.g. "You are a helpful assistant")
        - user_prompt: the actual task/question, usually including retrieved context
        - temperature: lower = more focused/deterministic answers (good for factual Q&A)
        """
        try:
            logger.info(f"Calling LLM ({self.model})...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            answer = response.choices[0].message.content
            logger.info("LLM responded successfully.")
            return answer

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fail gracefully with a clear message instead of crashing the app
            return (
                "⚠️ Sorry, I couldn't generate a response right now. "
                f"(Error: {e})"
            )
