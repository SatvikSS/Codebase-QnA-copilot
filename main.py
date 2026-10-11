from collections import Counter

from src.kb.chunker import chunk_file
from src.kb.config import load_config
from src.kb.discover import discover

config = load_config()
files = list(discover(config.projects_root, config.include_projects))

print("projects_root:", config.projects_root)
print("total files:", len(files))

print("\nfiles per project:")
for project, count in sorted(Counter(f.project for f in files).items()):
    print(f"  {count:6}  {project}")

print("\nfirst 5 files of each project:")
for project in sorted({f.project for f in files}):
    print(f"  {project}")
    for f in [f for f in files if f.project == project][:5]:
        print(f"    {f.rel_path}  ({f.size} bytes)")

chunks = [chunk for f in files for chunk in chunk_file(f)]
print("\ntotal chunks:", len(chunks))

print("\nchunks per project:")
for project, count in sorted(Counter(c.project for c in chunks).items()):
    print(f"  {count:6}  {project}")

print("\nsample chunks (first 2 of each project):")
for project in sorted({c.project for c in chunks}):
    print(f"  {project}")
    for c in [c for c in chunks if c.project == project][:2]:
        preview = c.text[:150].replace("\n", " | ")
        print(f"    {c.path}:{c.start_line}-{c.end_line}  ({len(c.text)} chars)")
        print(f"      {preview}")
