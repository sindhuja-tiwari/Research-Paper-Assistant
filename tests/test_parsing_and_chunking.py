from src.chunking import STRATEGIES, chunk_paper
from src.pdf_parser import ParsedPaper


def test_parser_finds_sections(parsed_paper):
    headings = [s.heading for s in parsed_paper.sections]
    assert "Abstract" in headings
    assert "1 Introduction" in headings
    assert "2 Results" in headings


def test_parser_keeps_table_rows_together(parsed_paper):
    assert len(parsed_paper.tables) == 1
    table = parsed_paper.tables[0]
    assert table.section == "2 Results"
    assert "94.1" in table.raw_text and "88.0" in table.raw_text


def test_parsed_paper_round_trips_through_json(parsed_paper, tmp_path):
    path = tmp_path / "paper.json"
    parsed_paper.to_json(path)
    loaded = ParsedPaper.from_json(path)
    assert loaded == parsed_paper


def test_all_strategies_produce_chunks(parsed_paper):
    for name in STRATEGIES:
        chunks = chunk_paper(parsed_paper, name)
        assert chunks, name
        assert all(c.arxiv_id == parsed_paper.arxiv_id for c in chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk ids must be unique"


def test_section_table_strategy_adds_table_chunk(parsed_paper):
    section = chunk_paper(parsed_paper, "section")
    section_table = chunk_paper(parsed_paper, "section_table")
    tables = [c for c in section_table if c.is_table]
    assert len(section_table) == len(section) + 1
    assert len(tables) == 1 and "94.1" in tables[0].text


def test_section_chunks_never_cross_sections(parsed_paper):
    for c in chunk_paper(parsed_paper, "section"):
        assert c.text.startswith(f"[{c.section}]")
