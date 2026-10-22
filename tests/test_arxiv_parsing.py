from src.arxiv_fetcher import parse_feed, safe_filename

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T17:42:21Z</published>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>  Large pre-trained language models
      have been shown to store factual knowledge. </summary>
    <author><name>Patrick Lewis</name></author>
    <author><name>Ethan Perez</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2005.11401v4" rel="related" type="application/pdf"/>
  </entry>
</feed>"""


def test_parse_feed():
    [paper] = parse_feed(FEED)
    assert paper.arxiv_id == "2005.11401v4"
    assert paper.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert paper.abstract.startswith("Large pre-trained language models have been")
    assert paper.authors == ["Patrick Lewis", "Ethan Perez"]
    assert paper.year() == 2020
    assert paper.pdf_url == "http://arxiv.org/pdf/2005.11401v4"


def test_safe_filename():
    assert safe_filename("2005.11401v4") == "2005.11401v4"
    assert safe_filename("hep-th/9901001v1") == "hep-th_9901001v1"
