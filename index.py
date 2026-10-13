import time

from src.kb.chunker import chunk_file
from src.kb.config import load_config
from src.kb.discover import discover
from src.kb.embedder import embed
from src.kb.manifest import file_hash, file_key, load_manifest, save_manifest
from src.kb.store import add_chunks, count, delete_file

start = time.time()
config = load_config()
files = {file_key(f): f for f in discover(config.projects_root, config.include_projects)}

old = load_manifest()
new = {key: file_hash(f) for key, f in files.items()}

deleted = [key for key in old if key not in new]
changed = [key for key in new if old.get(key) != new[key]]
print(f"files: {len(new)} | new or changed: {len(changed)} | deleted: {len(deleted)}")

# Remove old chunks first: an edited file may now have fewer chunks than before
for key in deleted + changed:
    project, path = key.split("/", 1)
    delete_file(project, path)

chunks = [chunk for key in changed for chunk in chunk_file(files[key])]
if chunks:
    add_chunks(chunks, embed([c.text for c in chunks]))

# Saved last: if the run crashes, the next run redoes the same work
save_manifest(new)
print(f"new chunks: {len(chunks)} | chunks in database: {count()} | {time.time() - start:.1f}s")
