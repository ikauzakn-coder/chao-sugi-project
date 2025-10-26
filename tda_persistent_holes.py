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
    ("scipy", None),
):
    ensure_package(package, import_name=import_name)


import numpy as np
from google.colab import files
from openai import OpenAI
from ripser import ripser
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_MODEL = "text-embedding-3-small"
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
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

    embeddings = model.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=False
    )
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
        "('paraphrase-multilingual-mpnet-base-v2')."
    )

    local_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    dimension = int(local_model.get_sentence_embedding_dimension())

    def embed(texts: Sequence[str]) -> np.ndarray:
        return embed_texts_sentence_transformer(local_model, texts, expected_dim=dimension)

    return "sentence-transformer", dimension, embed


def _cycle_vertices(cocycle: Iterable[Iterable[float]]) -> List[int]:
    """Return the sorted list of unique vertex indices participating in the cocycle."""

    vertices = set()
    for simplex in cocycle:
        for value in simplex:
            index = int(value)
            if index >= 0:
                vertices.add(index)
    return sorted(vertices)


def compute_cycle_center(
    embeddings: np.ndarray,
    cocycle: Iterable[Iterable[float]],
    birth: float,
    death: float,
    *,
    embedding_dim: int,
    homology_dim: int,
) -> np.ndarray:
    """Compute a representative center vector for a persistent 1-cycle.

    The center is estimated by averaging midpoints of edges in the cocycle, where each edge is
    weighted according to how close its length is to the midpoint between ``birth`` and ``death``.
    """

    if embeddings.size == 0:
        return np.zeros((embedding_dim,), dtype=np.float32)

    cocycle = list(cocycle)
    if not cocycle:
        return embeddings.mean(axis=0).astype(np.float32)

    midpoint_scale = max(1e-9, 0.25 * (birth + death))
    target_length = 0.5 * (birth + death)

    if homology_dim == 1:
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

        if edge_centers and not np.isclose(sum(weights), 0.0):
            weights_array = np.array(weights, dtype=np.float32).reshape(-1, 1)
            stacked_midpoints = np.vstack(edge_centers)
            weighted_center = (
                np.sum(stacked_midpoints * weights_array, axis=0) / np.sum(weights_array)
            )
            return weighted_center.astype(np.float32)

    elif homology_dim == 2:
        face_centers: List[np.ndarray] = []
        weights: List[float] = []

        for simplex in cocycle:
            vertices = [int(idx) for idx in simplex]
            if len(vertices) < 3:
                continue
            valid_vertices = [v for v in vertices if v < embeddings.shape[0]]
            if len(valid_vertices) < 3:
                continue

            simplex_points = embeddings[valid_vertices]
            centroid = simplex_points.mean(axis=0)

            pairwise = np.linalg.norm(
                simplex_points[:, None, :] - simplex_points[None, :, :], axis=-1
            )
            diameter = float(np.max(pairwise))
            weight = np.exp(-((diameter - target_length) ** 2) / (2.0 * midpoint_scale**2))

            face_centers.append(centroid)
            weights.append(weight)

        if face_centers and not np.isclose(sum(weights), 0.0):
            weights_array = np.array(weights, dtype=np.float32).reshape(-1, 1)
            stacked_centers = np.vstack(face_centers)
            weighted_center = (
                np.sum(stacked_centers * weights_array, axis=0) / np.sum(weights_array)
            )
            return weighted_center.astype(np.float32)

    vertices = _cycle_vertices(cocycle)
    if not vertices:
        return embeddings.mean(axis=0).astype(np.float32)
    valid_vertices = [v for v in vertices if v < embeddings.shape[0]]
    if not valid_vertices:
        return embeddings.mean(axis=0).astype(np.float32)
    return embeddings[valid_vertices].mean(axis=0).astype(np.float32)


def summarise_holes_for_dim(
    embeddings: np.ndarray,
    diagram: np.ndarray,
    cocycles: Sequence[Iterable[Iterable[float]]],
    word_embeddings: np.ndarray,
    candidate_words: Sequence[str],
    *,
    embedding_dim: int,
    homology_dim: int,
    top_n: int = TOP_N_LIFESPANS,
) -> List[Dict]:
    """Create summaries for the longest-lived holes of a specific dimension."""

    if word_embeddings.size == 0 or not candidate_words:
        return []

    if diagram is None or len(diagram) == 0:
        return []

    finite_entries = [
        (index, float(birth), float(death))
        for index, (birth, death) in enumerate(diagram)
        if np.isfinite(death)
    ]

    if not finite_entries:
        return []

    finite_entries.sort(key=lambda item: item[2] - item[1], reverse=True)
    top_entries = finite_entries[:top_n]

    summaries: List[Dict] = []
    for summary_id, (cycle_index, birth, death) in enumerate(top_entries, start=1):
        cocycle = cocycles[cycle_index] if cycle_index < len(cocycles) else []
        center_vector = compute_cycle_center(
            embeddings,
            cocycle,
            birth,
            death,
            embedding_dim=embedding_dim,
            homology_dim=homology_dim,
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
                "dim": homology_dim,
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

    poems = [
        # --- A: 自然系 ---
        "森の息がまだ残る夜明け、木々の影が風を縫う。",
        "光が葉脈を透かし、沈黙が形を持った。",
        "雨上がりの杉の匂いが、心の底を洗っていく。",
        "苔むす岩が、記憶よりも深く眠っている。",
        "川面を渡る風が、過去と未来を行き来する。",
        "雪解けの音が、森の時計を動かしている。",
        "陽だまりの中で、樹皮が言葉を覚え始めた。",
        "鳥の声が消えた瞬間、空がひと息ついた。",
        "種子が落ちる音を聴いた者だけが、森の名を知る。",
        "木漏れ日が目を閉じた心の奥に降りてくる。",
        # --- B: 人工系 ---
        "鉄の腕が静かに空を抱きしめた。",
        "AIが夢を見始めた夜、都市は眠ることを忘れた。",
        "機械の中で風が迷子になった。",
        "コードの中を光が走り、祈りがデータになる。",
        "電線が交差するたびに、記憶が書き換えられる。",
        "合金の心臓が鼓動を打つと、街が少しだけ呼吸した。",
        "プログラムが一行消え、世界の均衡が傾いた。",
        "ガラスの壁に触れると、未来が微かに震えた。",
        "鉄の花が咲くとき、人は言葉を思い出す。",
        "風のかわりにファンが歌い、森の夢を再現した。",
        # --- C: 人間性 ---
        "祈りは届かずとも、沈黙は返ってきた。",
        "時間は木の年輪のように、痛みを包み隠した。",
        "君の声が記憶をほどき、涙が再生を始めた。",
        "夢の中で私はまだ木を削っている。",
        "老いた手が光を掴もうとして震えた。",
        "命とは、忘却の森で響く最後の足音だ。",
        "心が折れた日、森がそっと息を合わせてくれた。",
        "技術の進歩が人の孤独を少しだけ延命した。",
        "風の中に人の声を探すことを、誰もやめられない。",
        "未来はきっと、木目の中に沈んでいる。",
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

    text_embeddings = embed(poems)
    word_embeddings = embed(candidate_words)

    if text_embeddings.shape[1] != word_embeddings.shape[1]:
        raise ValueError(
            "Text and candidate word embeddings must share the same dimensionality, "
            f"but got {text_embeddings.shape[1]} and {word_embeddings.shape[1]}."
        )

    from sklearn.metrics.pairwise import pairwise_distances

    # グループ重み付け（自然=1.0、人工=1.5、人間性=2.0）
    group_weights = np.array([1.0]*10 + [1.5]*10 + [2.0]*10, dtype=np.float32).reshape(-1, 1)
    text_embeddings = text_embeddings * group_weights

    # cosine距離行列を使ってRipser解析
    distance_matrix = pairwise_distances(text_embeddings, metric="cosine")
    persistence_result = ripser(distance_matrix, distance_matrix=True, maxdim=3, do_cocycles=True)


    diagrams = persistence_result.get("dgms", [])
    cocycles = persistence_result.get("cocycles", [])

    print(f"Poems embedded: {len(poems)}")
    print(f"Embedding dimension: {text_embeddings.shape[1]}")
    for dim, diag in enumerate(diagrams):
        finite = np.sum(np.isfinite(diag[:, 1])) if diag.size else 0
        print(f"H{dim} bars (finite): {int(finite)}")

    h1_diagram = diagrams[1] if len(diagrams) > 1 else None
    h1_cocycles = cocycles[1] if len(cocycles) > 1 else []
    h2_diagram = diagrams[2] if len(diagrams) > 2 else None
    h2_cocycles = cocycles[2] if len(cocycles) > 2 else []

    h1_total = len(h1_diagram) if h1_diagram is not None else 0
    h1_finite = int(np.sum(np.isfinite(h1_diagram[:, 1]))) if h1_diagram is not None else 0
    h2_total = len(h2_diagram) if h2_diagram is not None else 0
    h2_finite = int(np.sum(np.isfinite(h2_diagram[:, 1]))) if h2_diagram is not None else 0

    print(f"H1 bars (total/finite): {h1_total}/{h1_finite}")
    print(f"H2 bars (total/finite): {h2_total}/{h2_finite}")

    h1_holes = summarise_holes_for_dim(
        text_embeddings,
        h1_diagram,
        h1_cocycles,
        word_embeddings,
        candidate_words,
        embedding_dim=embedding_dim,
        homology_dim=1,
    )

    h2_holes = summarise_holes_for_dim(
        text_embeddings,
        h2_diagram,
        h2_cocycles,
        word_embeddings,
        candidate_words,
        embedding_dim=embedding_dim,
        homology_dim=2,
    )

    holes = h1_holes + h2_holes

    results = {"holes": holes}

    output_path = "tda_hole_results.json"
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    if not h1_holes:
        print("H₁の長寿命な穴は見つかりませんでした。")
    if not h2_holes:
        print("H₂の長寿命な穴は見つかりませんでした。")

    files.download(output_path)


if __name__ == "__main__":
    main()
