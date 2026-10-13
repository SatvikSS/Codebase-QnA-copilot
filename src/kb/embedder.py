"""Turn text into vectors with a local model."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

_model = SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = _model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    )
    return vectors.tolist()
