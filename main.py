"""
MSCP - Multi-System Compositional Pipeline
==========================================
main.py — The orchestration loop

Flow:
  User Input → Layer 4 (Memory) → Layer 3 (Graph) → Layer 2 (FAISS)
             → Layer 5 (Assembler) → Layer 1 (LLM) → Streamed Response

Commands:
  /graph     - Visualize the concept graph
  /memory    - Show current short-term buffer
  /ingest    - Re-ingest knowledge_base/ folder
  /clear     - Clear short-term memory
  /help      - Show commands
  /exit      - Quit
"""

import sys
import os

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

def color(text, c=""):
    if HAS_COLOR:
        return c + text + Style.RESET_ALL
    return text

def print_banner():
    banner = r"""
  __  __ ____   ____ ____  
 |  \/  / ___| / ___|  _ \ 
 | |\/| \___ \| |   | |_) |
 | |  | |___) | |___|  __/ 
 |_|  |_|____/ \____|_|    
  Multi-System Compositional Pipeline
    """
    if HAS_COLOR:
        print(Fore.CYAN + banner + Style.RESET_ALL)
    else:
        print(banner)
    print("  Type /help for commands. Type /exit to quit.\n")

def show_help():
    cmds = [
        ("/graph",   "Visualize the concept graph (saves concept_graph.png)"),
        ("/memory",  "Show current short-term chat buffer"),
        ("/ingest",  "Re-scan knowledge_base/ and add new .txt files"),
        ("/clear",   "Clear short-term memory buffer"),
        ("/help",    "Show this help"),
        ("/exit",    "Quit MSCP"),
    ]
    print(color("\n  Available Commands:", Fore.YELLOW if HAS_COLOR else ""))
    for cmd, desc in cmds:
        print(f"    {color(cmd, Fore.GREEN if HAS_COLOR else '')}  —  {desc}")
    print()

def main():
    print_banner()

    # ── Initialize all layers ──────────────────────────────────
    print(color("[MSCP] Initializing layers...\n", Fore.YELLOW if HAS_COLOR else ""))

    # Layer 2: Vector Brain (must init first, others depend on it)
    from layer2_vector import VectorBrain
    vector_brain = VectorBrain()

    # Layer 3: Concept Graph
    from layer3_graph import ConceptGraph
    concept_graph = ConceptGraph()

    # Layer 4: Short-Term Memory Buffer
    from layer4_memory import ShortTermBuffer
    memory_buffer = ShortTermBuffer(vector_brain)

    # Layer 5: Prompt Assembler (stateless functions, no init needed)
    from layer5_assembler import assemble_prompt, count_tokens_approx

    # Layer 1: LLM Engine (loads the model — takes a moment)
    from layer1_engine import LLMEngine
    llm = LLMEngine()

    print(color("\n[MSCP] All systems online. Ready!\n", Fore.GREEN if HAS_COLOR else ""))
    print("-" * 60)

    # ── Main Loop ─────────────────────────────────────────────
    while True:
        try:
            if HAS_COLOR:
                user_input = input(Fore.CYAN + "\nYou: " + Style.RESET_ALL).strip()
            else:
                user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[MSCP] Goodbye!")
            break

        if not user_input:
            continue

        # ── Handle commands ────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd == "/exit":
                print("[MSCP] Goodbye!")
                break

            elif cmd == "/help":
                show_help()

            elif cmd == "/graph":
                concept_graph.visualize()

            elif cmd == "/memory":
                history = memory_buffer.format_for_prompt()
                print(color("\n[Layer 4 - Short-Term Buffer]:", Fore.YELLOW if HAS_COLOR else ""))
                print(history)

            elif cmd == "/clear":
                memory_buffer.clear()
                print("[Layer 4] Short-term memory cleared.")

            elif cmd == "/ingest":
                print("[Layer 2] Re-ingesting knowledge_base/ ...")
                vector_brain._ingest_knowledge_base()

            else:
                print(f"Unknown command: {user_input}. Type /help for commands.")
            continue

        # ── Process normal user message ────────────────────────

        # Step 1: Update memory buffer with user message
        memory_buffer.add("user", user_input)

        # Step 2: Layer 3 — Get related concepts from graph
        graph_concepts = concept_graph.retrieve(user_input)

        # Step 3: Layer 2 — Retrieve relevant FAISS chunks
        faiss_chunks = vector_brain.retrieve(user_input, top_k=5)

        # Step 4: Layer 4 — Get formatted chat history
        chat_history = memory_buffer.format_for_prompt()

        # Step 5: Layer 5 — Assemble the full prompt
        prompt = assemble_prompt(
            user_input=user_input,
            graph_concepts=graph_concepts,
            faiss_chunks=faiss_chunks,
            chat_history=chat_history,
            temperature_hint=0.2
        )

        # Show a small debug hint (optional)
        token_estimate = count_tokens_approx(prompt)
        if HAS_COLOR:
            print(Fore.YELLOW + f"[↑ Ctx: ~{token_estimate} tokens | Graph: {graph_concepts} | FAISS: {len(faiss_chunks)} chunks]" + Style.RESET_ALL)
        else:
            print(f"[Ctx: ~{token_estimate} tokens | Graph: {graph_concepts} | FAISS: {len(faiss_chunks)} chunks]")

        # Step 6: Layer 3 — Update graph with new input
        concept_graph.update(user_input)

        # Step 7: Layer 1 — Stream the LLM response
        if HAS_COLOR:
            print(Fore.GREEN + "\nAssistant: " + Style.RESET_ALL, end="", flush=True)
        else:
            print("\nAssistant: ", end="", flush=True)

        full_response = ""
        try:
            for token in llm.generate(prompt, max_tokens=512, temperature=0.2, stream=True):
                print(token, end="", flush=True)
                full_response += token
        except Exception as e:
            print(f"\n[Layer 1 Error] {e}")

        print()  # Newline after streamed response

        # Step 8: Save assistant response to memory buffer
        if full_response.strip():
            memory_buffer.add("assistant", full_response.strip())

        print("-" * 60)


if __name__ == "__main__":
    main()
