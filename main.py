"""Interactive CLI for the LangGraph + DSPy banking assistant.

Run:
    gcloud auth application-default login   # one-time, for Vertex AI ADC
    python main.py                          # reads .env for project/location/model

Try (as test customer 'Ada Lovelace'):
    "what's my balance?"            -> triggers auth
    ssn last 4: 4321
    dob: 1985-06-15
    zip: 10001
    -> verified, then balance is shown. Then ask "show my last transactions".
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from src.config import configure_dspy
from src.graph import build_graph

QUIT = {"quit", "exit", "bye", "q"}


def _print_new_agent_messages(result: dict, seen: int) -> int:
    """Print every AIMessage added since the last turn, return the new count.

    A single turn can emit more than one assistant message (e.g. the auth node
    says "identity is verified." and then the balance node responds in the same
    invoke). ``response`` only holds the *last* one, so we render the full
    ``messages`` tail instead to avoid dropping intermediate replies.
    """
    messages = result.get("messages", [])
    for message in messages[seen:]:
        if isinstance(message, AIMessage):
            print(f"\nAgent: {message.content}\n")
    return len(messages)


def main() -> None:
    configure_dspy()
    graph = build_graph()
    thread = {"configurable": {"thread_id": f"session-{uuid.uuid4().hex[:8]}"}}

    print("=" * 64)
    print("  Banking Assistant  (LangGraph orchestration + DSPy prompting)")
    print("=" * 64)
    print("Type 'quit' to exit.\n")

    # Kick off with an empty turn so the assistant greets and asks for language.
    greeting = graph.invoke({"messages": []}, thread)
    seen = _print_new_agent_messages(greeting, 0)

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user:
            continue
        if user.lower() in QUIT:
            print("Goodbye!")
            break

        result = graph.invoke({"messages": [HumanMessage(content=user)]}, thread)
        seen = _print_new_agent_messages(result, seen)

        # The assistant ended the conversation (customer said they're done).
        if result.get("call_ended"):
            break


if __name__ == "__main__":
    main()
