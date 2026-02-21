"""
LAYER 5: Prompt Assembly & Personality Injector
-------------------------------------------------
- Combines all retrieved context into a single, structured prompt
- Injects the AI's persona and behavior settings
- Formats output for the Llama 3.2 instruction format

FIX: Do NOT include <|begin_of_text|> — llama.cpp adds it automatically.
     Including it manually causes the duplicate token warning and breaks responses.
"""

SYSTEM_PERSONA = """You are a helpful, knowledgeable assistant.
Answer questions directly in plain text. Do NOT write Python code unless the user explicitly asks for code.
Use the RETRIEVED KNOWLEDGE below to answer factual questions accurately.
If the knowledge contains the answer, use it. Be concise and direct."""

def assemble_prompt(
    user_input: str,
    graph_concepts: list,
    faiss_chunks: list,
    chat_history: str,
    temperature_hint: float = 0.2
) -> str:
    """
    Assemble the full prompt from all 5 layers.

    IMPORTANT: No <|begin_of_text|> here — llama.cpp adds it automatically.
    Including it causes duplicate token warning and completely breaks the model output.
    """

    # --- Format FAISS chunks (full text, not truncated) ---
    if faiss_chunks:
        knowledge_block = "\n\n".join(f"[FACT {i+1}]: {chunk}" for i, chunk in enumerate(faiss_chunks))
        knowledge_section = f"RETRIEVED KNOWLEDGE:\n{knowledge_block}"
    else:
        knowledge_section = ""

    # --- Format Graph concepts ---
    if graph_concepts:
        concept_section = f"RELATED TOPICS: {', '.join(graph_concepts)}"
    else:
        concept_section = ""

    # --- Format chat history ---
    if chat_history and chat_history != "No previous messages.":
        history_section = f"CONVERSATION HISTORY:\n{chat_history}"
    else:
        history_section = ""

    # --- Build ONE system block ---
    system_parts = [SYSTEM_PERSONA]
    if knowledge_section:
        system_parts.append(knowledge_section)
    if concept_section:
        system_parts.append(concept_section)
    if history_section:
        system_parts.append(history_section)

    system_block = "\n\n".join(system_parts)

    # --- Llama 3.2 instruct format — NO <|begin_of_text|> ---
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_block}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return prompt


def count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4
