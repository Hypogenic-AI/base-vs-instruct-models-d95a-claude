import arxiv, json, sys
queries = sys.argv[1:]
client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=3)
seen=set()
out=[]
for q in queries:
    search = arxiv.Search(query=q, max_results=12, sort_by=arxiv.SortCriterion.Relevance)
    try:
        for r in client.results(search):
            aid = r.entry_id.split('/abs/')[-1]
            if aid in seen: continue
            seen.add(aid)
            out.append({"q":q,"id":aid,"title":r.title.replace('\n',' '),
                        "year":r.published.year,"authors":[a.name for a in r.authors][:4],
                        "summary":r.summary.replace('\n',' ')[:600],"pdf":r.pdf_url})
    except Exception as e:
        print("ERR",q,e,file=sys.stderr)
print(json.dumps(out,indent=1))
