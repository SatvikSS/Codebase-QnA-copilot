import time

from src.kb import keyword, store
from src.kb.chunker import chunk_file, with_header
from src.kb.config import load_config
from src.kb.discover import discover
from src.kb.embedder import embed
from src.kb.manifest import file_hash, file_key, load_manifest, save_manifest

start = time.time()
config = load_config()
files = {file_key(f): f for f in discover(config.projects_root, config.include_projects)}

old = load_manifest()
new = {key: file_hash(f) for key, f in files.items()}

deleted = [key for key in old if key not in new]
changed = [key for key in new if old.get(key) != new[key]]
print(f"files: {len(new)} | new or changed: {len(changed)} | deleted: {len(deleted)}")

# Remove old chunks first: an edited file may now have fewer chunks than before.
# A file that was never indexed has nothing to remove.
for key in deleted + [key for key in changed if key in old]:
    project, path = key.split("/", 1)
    store.delete_file(project, path)
    keyword.delete_file(project, path)

chunks = [chunk for key in changed for chunk in chunk_file(files[key])]
if chunks:
    store.add_chunks(chunks, embed([with_header(c) for c in chunks]))
    keyword.add_chunks(chunks)

# Saved last: if the run crashes, the next run redoes the same work
save_manifest(new)
print(f"new chunks: {len(chunks)} | chunks in database: {store.count()} | {time.time() - start:.1f}s")
