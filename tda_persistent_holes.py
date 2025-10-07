"""Script for computing persistent homology on text embeddings and exporting hole summaries."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


def ensure_package(package_name: str, *, import_name: str | None = None) -> None:
    """Install ``package_name`` with pip if the import target is unavailable."""

    target = import_name or package_name
    if importlib.util.find_spec(target) is None:
        print(f"Installing {package_name}…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


for package, import_name in (
    ("openai", None),
    ("ripser", None),
    ("persim", None),
    ("scikit-learn", "sklearn"),
    ("sentence-transformers", "sentence_transformers"),
):
    ensure_package(package, import_name=import_name)


import numpy as np
from google.colab import files
from openai import OpenAI
from ripser import ripser
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_MODEL = "text-embedding-3-small"
TOP_N_LIFESPANS = 3


def embed_texts_openai(
    client: OpenAI,
    texts: Sequence[str],
    *,
    model: str = EMBEDDING_MODEL,
    expected_dim: int = 1536,
) -> np.ndarray:
    """Embed ``texts`` using the OpenAI API."""

    if not texts:
        return np.empty((0, expected_dim), dtype=np.float32)

    response = client.embeddings.create(model=model, input=list(texts))
    vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]

    if not vectors:
        return np.empty((0, expected_dim), dtype=np.float32)

    stacked = np.vstack(vectors)
    if stacked.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected embedding dimension {stacked.shape[1]} (expected {expected_dim})."
        )

    return stacked


def embed_texts_sentence_transformer(
    model, texts: Sequence[str], *, expected_dim: int
) -> np.ndarray:
    """Embed ``texts`` using a SentenceTransformer ``model``."""

    if not texts:
        return np.empty((0, expected_dim), dtype=np.float32)

    embeddings = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected embedding dimension {embeddings.shape[1]} (expected {expected_dim})."
        )

    return embeddings.astype(np.float32)


def resolve_embedding_backend() -> Tuple[str, int, Callable[[Sequence[str]], np.ndarray]]:
    """Choose an embedding backend and return (name, dimension, callable)."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)

        def embed(texts: Sequence[str]) -> np.ndarray:
            return embed_texts_openai(client, texts, expected_dim=1536)

        return "openai", 1536, embed

    from sentence_transformers import SentenceTransformer

    print(
        "OPENAI_API_KEY not set. Falling back to a local SentenceTransformer model "
        "('paraphrase-multilingual-MiniLM-L12-v2')."
    )

    local_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    dimension = int(local_model.get_sentence_embedding_dimension())

    def embed(texts: Sequence[str]) -> np.ndarray:
        return embed_texts_sentence_transformer(local_model, texts, expected_dim=dimension)

    return "sentence-transformer", dimension, embed


def _cycle_vertices(cocycle: Iterable[Iterable[float]]) -> List[int]:
    """Return the sorted list of unique vertex indices participating in the cocycle."""

    vertices = set()
    for simplex in cocycle:
        if len(simplex) < 2:
            continue
        i, j = int(simplex[0]), int(simplex[1])
        vertices.add(i)
        vertices.add(j)
    return sorted(vertices)


def compute_cycle_center(
    embeddings: np.ndarray,
    cocycle: Iterable[Iterable[float]],
    birth: float,
    death: float,
    *,
    embedding_dim: int,
) -> np.ndarray:
    """Compute a representative center vector for a persistent 1-cycle.

    The center is estimated by averaging midpoints of edges in the cocycle, where each edge is
    weighted according to how close its length is to the midpoint between ``birth`` and ``death``.
    """

    if embeddings.size == 0:
        return np.zeros((embedding_dim,), dtype=np.float32)

    cocycle = list(cocycle)
    if not cocycle:
        return embeddings.mean(axis=0)

    midpoint_scale = max(1e-9, 0.25 * (birth + death))
    target_length = 0.5 * (birth + death)

    edge_centers: List[np.ndarray] = []
    weights: List[float] = []

    for simplex in cocycle:
        if len(simplex) < 2:
            continue
        i, j = int(simplex[0]), int(simplex[1])
        if i >= embeddings.shape[0] or j >= embeddings.shape[0]:
            continue

        point_i = embeddings[i]
        point_j = embeddings[j]
        midpoint = 0.5 * (point_i + point_j)
        edge_length = float(np.linalg.norm(point_i - point_j))

        weight = np.exp(-((edge_length - target_length) ** 2) / (2.0 * midpoint_scale**2))
        edge_centers.append(midpoint)
        weights.append(weight)

    if not edge_centers or np.isclose(sum(weights), 0.0):
        vertices = _cycle_vertices(cocycle)
        if not vertices:
            return embeddings.mean(axis=0)
        return embeddings[vertices].mean(axis=0)

    weights_array = np.array(weights, dtype=np.float32).reshape(-1, 1)
    stacked_midpoints = np.vstack(edge_centers)
    weighted_center = np.sum(stacked_midpoints * weights_array, axis=0) / np.sum(weights_array)
    return weighted_center.astype(np.float32)


def summarise_holes(
    embeddings: np.ndarray,
    persistence_result: Dict[str, List[np.ndarray]],
    word_embeddings: np.ndarray,
    candidate_words: Sequence[str],
    *,
    embedding_dim: int,
    top_n: int = TOP_N_LIFESPANS,
) -> List[Dict]:
    """Create summaries for the longest-lived H1 holes."""

    if word_embeddings.size == 0 or not candidate_words:
        return []

    diagrams = persistence_result.get("dgms", [])
    cocycles = persistence_result.get("cocycles", [])
    if len(diagrams) < 2 or len(cocycles) < 2:
        return []

    h1_diagram = diagrams[1]
    h1_cocycles = cocycles[1]

    if h1_diagram.size == 0:
        return []

    finite_entries = [
        (index, float(birth), float(death))
        for index, (birth, death) in enumerate(h1_diagram)
        if np.isfinite(death)
    ]

    if not finite_entries:
        return []

    finite_entries.sort(key=lambda item: item[2] - item[1], reverse=True)
    top_entries = finite_entries[:top_n]

    summaries: List[Dict] = []
    for summary_id, (cycle_index, birth, death) in enumerate(top_entries, start=1):
        cocycle = h1_cocycles[cycle_index] if cycle_index < len(h1_cocycles) else []
        center_vector = compute_cycle_center(
            embeddings, cocycle, birth, death, embedding_dim=embedding_dim
        )
        similarities = cosine_similarity(center_vector.reshape(1, -1), word_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1][:5]

        top_words = [
            {
                "word": candidate_words[idx],
                "similarity": float(similarities[idx]),
            }
            for idx in ranked_indices
        ]

        summaries.append(
            {
                "id": summary_id,
                "birth": birth,
                "death": death,
                "lifespan": death - birth,
                "center_vector": center_vector.astype(float).tolist(),
                "top_words": top_words,
            }
        )

    return summaries


def main() -> None:
    backend_name, embedding_dim, embed = resolve_embedding_backend()

    if backend_name == "openai":
        print(
            "Using OpenAI embeddings (model: text-embedding-3-small). Ensure API usage complies with your plan."
        )
    else:
        print(
            "Using SentenceTransformer embeddings; results may differ from OpenAI's 1536-dimensional vectors."
        )

    texts = [
        "木々の間を抜ける風が、忘れられた歌を運んでくる。",
        "錆びた機械が祈りのように静かに息をしている。",
        "記憶の底で光る青い石は、誰の夢だったのだろう。",
        "沈黙の中、AIが新しい言葉を紡ぎ出す夜。",
        "再生する森とともに、時間の波が静かに満ちていく。",
    ]

    candidate_words = [
        "木",
        "森",
        "機械",
        "祈り",
        "記憶",
        "呼吸",
        "鉄",
        "光",
        "沈黙",
        "AI",
        "風",
        "命",
        "時間",
        "自然",
        "再生",
        "夢",
        "技術",
        "音",
        "人間",
    ]

    text_embeddings = embed(texts)
    word_embeddings = embed(candidate_words)

    if text_embeddings.shape[1] != word_embeddings.shape[1]:
        raise ValueError(
            "Text and candidate word embeddings must share the same dimensionality, "
            f"but got {text_embeddings.shape[1]} and {word_embeddings.shape[1]}."
        )

    persistence_result = ripser(text_embeddings, maxdim=1, do_cocycles=True)
    holes = summarise_holes(
        text_embeddings,
        persistence_result,
        word_embeddings,
        candidate_words,
        embedding_dim=embedding_dim,
    )

    results = {"holes": holes}

    output_path = "tda_hole_results.json"
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    if not holes:
        print("長寿命な穴は見つかりませんでした。")

    files.download(output_path)


if __name__ == "__main__":
    main()
