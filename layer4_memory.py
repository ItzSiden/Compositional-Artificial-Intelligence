"""
LAYER 4: Short-Term Memory Buffer
-----------------------------------
- Holds the last 5 messages using collections.deque
- When a 6th message arrives, the oldest is compressed by the LLM
  and saved to FAISS (Layer 2) as a long-term memory
- This prevents context overflow while preserving important knowledge
"""

from collections import deque

MAX_BUFFER = 5

class ShortTermBuffer:
    def __init__(self, vector_brain):
        self.buffer = deque(maxlen=MAX_BUFFER)
        self.vector_brain = vector_brain
        self.overflow_count = 0
        print(f"[Layer 4] Short-term memory buffer initialized (capacity: {MAX_BUFFER} messages).")

    def add(self, role: str, content: str):
        """Add a message. If buffer is full, the oldest is evicted to long-term memory."""
        if len(self.buffer) == MAX_BUFFER:
            # The oldest message is about to be pushed out — compress it into FAISS
            oldest = self.buffer[0]
            self._compress_to_long_term(oldest)

        self.buffer.append({"role": role, "content": content})

    def _compress_to_long_term(self, message: dict):
        """
        Summarize and save the evicted message to FAISS long-term memory.
        Note: We save the raw text as a memory chunk. In a full system, you'd
        call the LLM here to summarize it first, but to keep the pipeline simple
        we directly store the content with a tag.
        """
        role = message["role"]
        content = message["content"]
        memory = f"[PAST MEMORY - {role.upper()}]: {content[:300]}"  # Truncate for brevity
        self.vector_brain.add_and_save(memory)
        self.overflow_count += 1

    def get_history(self) -> list:
        """Return the current buffer as a list of dicts."""
        return list(self.buffer)

    def format_for_prompt(self) -> str:
        """Format the chat history as a readable string block."""
        if not self.buffer:
            return "No previous messages."
        lines = []
        for msg in self.buffer:
            role = msg["role"].capitalize()
            content = msg["content"][:200]  # Trim long messages
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self):
        self.buffer.clear()
