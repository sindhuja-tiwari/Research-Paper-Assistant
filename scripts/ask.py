"""
Interactive CLI for the research agent. This is your demo script --
run it live in an interview and ask a cross-paper comparison question
to show the multi-step tool trace.

Usage:
    python -m scripts.ask
    python -m scripts.ask --query "Compare how paper X and paper Y handle long context"
"""
import argparse
import sys

from openai import OpenAIError

from src.agent import ResearchAgent, print_trace


def main(strategy: str, use_reranker: bool, query: str = None):
    try:
        agent = ResearchAgent(strategy=strategy, use_reranker=use_reranker)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if query:
        _ask(agent, query)
        return

    print("Research Paper Assistant (type 'quit' to exit)")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        _ask(agent, q)


def _ask(agent: ResearchAgent, query: str):
    try:
        print_trace(agent.run(query))
    except OpenAIError as e:
        print(f"OpenAI request failed: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n(cancelled)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="section_table",
                         choices=["naive", "section", "section_table"])
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()
    main(args.strategy, use_reranker=not args.no_rerank, query=args.query)
