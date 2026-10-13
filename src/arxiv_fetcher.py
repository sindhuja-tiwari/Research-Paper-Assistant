def search_arxiv(query: str, max_results: int = 10) -> List[ArxivPaper]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "RAG-Paper-Assistant/1.0 (research paper discovery)"
    }

    # Retry only transient errors. Do not hammer the API on HTTP 429.
    for attempt in range(3):
        try:
            resp = requests.get(
                config.ARXIV_API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 30 * (attempt + 1)
                except ValueError:
                    delay = 30 * (attempt + 1)

                if attempt == 2:
                    resp.raise_for_status()

                delay = min(max(delay, 1), 120)
                print(
                    f"arXiv rate limit (429). "
                    f"Waiting {delay:g}s before retry {attempt + 2}/3..."
                )
                time.sleep(delay)
                continue

            if resp.status_code in (500, 502, 503, 504):
                if attempt == 2:
                    resp.raise_for_status()
                delay = 5 * (attempt + 1)
                print(f"arXiv server error ({resp.status_code}); retrying in {delay}s...")
                time.sleep(delay)
                continue

            resp.raise_for_status()
            break

        except requests.RequestException:
            if attempt == 2:
                raise
            delay = 5 * (attempt + 1)
            print(f"arXiv request failed; retrying in {delay}s...")
            time.sleep(delay)
    else:
        raise RuntimeError("arXiv request failed after retries")

    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        raw_id = entry.find("atom:id", ARXIV_NS).text.strip()
        arxiv_id = raw_id.split("/abs/")[-1]
        title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ARXIV_NS).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ARXIV_NS).text[:10]
        authors = [
            a.find("atom:name", ARXIV_NS).text
            for a in entry.findall("atom:author", ARXIV_NS)
        ]

        pdf_url = None
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")

        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                pdf_url=pdf_url,
            )
        )

    return papers
