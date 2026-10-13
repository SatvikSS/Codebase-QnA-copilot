from src.kb.chunker import chunk_file
from src.kb.config import load_config
from src.kb.discover import discover
from src.kb.embedder import embed
from src.kb.store import add_chunks, count

config = load_config()
files = discover(config.projects_root, config.include_projects)
chunks = [chunk for f in files for chunk in chunk_file(f)]
print("chunks to index:", len(chunks))

vectors = embed([c.text for c in chunks])
add_chunks(chunks, vectors)
print("chunks in database:", count())
