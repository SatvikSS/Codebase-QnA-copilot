import sys

from src.kb.embedder import embed
from src.kb.store import search

question = " ".join(sys.argv[1:])
[query_vector] = embed([question])

for hit in search(query_vector, k=5):
    print(f"{hit['project']}/{hit['path']}:{hit['start_line']}-{hit['end_line']}  (distance {hit['distance']:.3f})")
    print("   ", hit["text"][:120].replace("\n", " | "))
