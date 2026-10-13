import sys

from src.kb.retrieval import search_chunks

question = " ".join(sys.argv[1:])

for hit in search_chunks(question, k=5):
    print(f"{hit['project']}/{hit['path']}:{hit['start_line']}-{hit['end_line']}  (score {hit['score']:.4f})")
    print("   ", hit["text"][:120].replace("\n", " | "))
