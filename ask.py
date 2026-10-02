"""
Interactive CLI for the research agent. This is your demo script --
run it live in an interview and ask a cross-paper comparison question
to show the multi-step tool trace.

Usage:
    python -m scripts.ask
    python -m scripts.ask --query "Compare how paper X and paper Y handle long context"
"""
import argparse

from src.agent import ResearchAgent, print_trace


def main(strategy: str, use_reranker: bool, query: str = None):
    agent = ResearchAgent(strategy=strategy, use_reranker=use_reranker)

    if query:
        trace = agent.run(query)
        print_trace(trace)
        return

    print("Research Paper Assistant (type 'quit' to exit)")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        trace = agent.run(q)
        print_trace(trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="section_table",
                         choices=["naive", "section", "section_table"])
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()
    main(args.strategy, use_reranker=not args.no_rerank, query=args.query)
