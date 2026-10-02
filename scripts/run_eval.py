"""
Runs the full retrieval evaluation and prints a markdown table you can
paste straight into your README.

Usage:
    python -m scripts.run_eval
"""
from src.eval import run_full_comparison, print_markdown_table

if __name__ == "__main__":
    results = run_full_comparison()
    print("\n--- Markdown table (paste into README) ---\n")
    print_markdown_table(results)
