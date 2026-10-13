from src.kb.config import load_config
from src.kb.indexer import index_projects

config = load_config()
print(index_projects(config.projects_root, config.include_projects))
