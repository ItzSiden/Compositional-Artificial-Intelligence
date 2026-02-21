"""
LAYER 5: Prompt Assembly & Personality Injector
-------------------------------------------------
- Combines all retrieved context into a single, structured prompt
- Injects the AI's persona and behavior settings
- Formats output for the Llama 3.2 instruction format
"""

SYSTEM_PERSONA = """You are a concise, expert coder and technical assistant. 
You give precise, well-formatted answers. You avoid unnecessary commentary.
When writing code, always include comments. You think step by step."""

def assemble_prompt(
    user_input: str,
    graph_concepts: list,
    faiss_chunks: list,
    chat_history: str,
    temperature_hint: float = 0.2
) -> str:
    """
    Assemble the full prompt from all 5 layers.

    Structure:
    [SYSTEM PERSONA]
    [LONG-TERM KNOWLEDGE from FAISS]
    [RELATED CONCEPTS from Graph]
    [SHORT-TERM CHAT HISTORY]
    [CURRENT USER QUESTION]
    """

    # --- Format FAISS chunks ---
    if faiss_chunks:
        knowledge_block = "\n".join(f"- {chunk[:250]}" for chunk in faiss_chunks)
        knowledge_section = f"""<knowledge>
Relevant information retrieved from memory:
{knowledge_block}
</knowledge>"""
    else:
        knowledge_section = ""

    # --- Format Graph concepts ---
    if graph_concepts:
        concepts_str = ", ".join(graph_concepts)
        concept_section = f"<related_concepts>Strongly related topics: {concepts_str}</related_concepts>"
    else:
        concept_section = ""

    # --- Format chat history ---
    history_section = f"""<chat_history>
{chat_history}
</chat_history>""" if chat_history and chat_history != "No previous messages." else ""

    # --- Build the system block (all context merged into ONE system message) ---
    system_parts = [SYSTEM_PERSONA]

    if knowledge_section:
        system_parts.append(knowledge_section)
    if concept_section:
        system_parts.append(concept_section)
    if history_section:
        system_parts.append(history_section)

    system_block = "\n\n".join(system_parts)

    # --- Llama 3.2 instruct format (single <|begin_of_text|>, one system message) ---
    prompt = (
        f"<|begin_of_text|>"
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
