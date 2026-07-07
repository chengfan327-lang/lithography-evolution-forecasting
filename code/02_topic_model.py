"""
Sentence-BERT-KMeans topic modeling for lithography patent texts.

This script identifies technological themes from patent titles and abstracts using
Sentence-BERT embeddings, UMAP dimensionality reduction, K-Means clustering, and
BERTopic-based topic representation. It is intended for reproducible research while
avoiding the redistribution of licensed raw patent records.

Example:
    python code/02_topic_model_open_source.py \
        --input data/lithography_cleaned.xlsx \
        --model-path /path/to/PatentSBERTa \
        --output-dir outputs/topic_model \
        --n-topics 17
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

warnings.filterwarnings("ignore")

# Domain-specific high-frequency terms that are weakly informative in patent texts.
PATENT_STOPWORDS = [
    "method", "device", "apparatus", "system", "invention", "present", "disclosed",
    "includes", "comprising", "provided", "step", "process", "use", "using",
    "thereof", "therein", "herein", "embodiment", "example", "configured",
    "manufacturing", "manufacture", "forming", "formed", "problem", "solution",
    "technique", "related", "relates", "layer", "substrate", "structure",
    "based", "having", "according", "second", "first", "third", "plurality",
]


def log(message: str, level: str = "INFO") -> None:
    """Print a timestamped log message."""
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def clean_patent_text(text: str) -> str:
    """Normalize patent text by lowercasing and retaining alphabetic tokens only."""
    text = str(text).lower()
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def read_table(path: Path) -> pd.DataFrame:
    """Read an Excel or CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input file must be .xlsx, .xls, or .csv")


def build_documents(
    df: pd.DataFrame,
    title_col: str,
    abstract_col: str,
) -> pd.Series:
    """Combine title and abstract columns into cleaned patent documents."""
    missing_cols = [col for col in [title_col, abstract_col] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required column(s): {missing_cols}")

    combined = df[title_col].fillna("") + " " + df[abstract_col].fillna("")
    return combined.map(clean_patent_text)


def save_parameters(args: argparse.Namespace, output_dir: Path) -> None:
    """Save model and preprocessing parameters for reproducibility."""
    params = vars(args).copy()
    params["run_time"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(output_dir / "topic_model_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sentence-BERT-KMeans topic modeling for lithography patent texts."
    )
    parser.add_argument("--input", required=True, help="Path to the input patent file (.xlsx or .csv).")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path or Hugging Face identifier of the Sentence-BERT/PatentSBERTa model.",
    )
    parser.add_argument("--output-dir", default="outputs/topic_model", help="Directory for output files.")
    parser.add_argument("--title-col", default="Title", help="Column name for patent titles.")
    parser.add_argument("--abstract-col", default="Abstract", help="Column name for patent abstracts.")
    parser.add_argument("--id-col", default=None, help="Optional document or patent ID column.")
    parser.add_argument("--n-topics", type=int, default=17, help="Number of K-Means clusters/topics.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding generation.")
    parser.add_argument("--min-df", type=int, default=10, help="Minimum document frequency for topic words.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="Number of UMAP neighbors.")
    parser.add_argument("--umap-components", type=int, default=5, help="Number of UMAP dimensions.")
    parser.add_argument("--mmr-diversity", type=float, default=0.3, help="MMR diversity parameter.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--include-text-output",
        action="store_true",
        help=(
            "Include original title and abstract columns in documents_per_topic.csv. "
            "Do not enable this option if raw patent records cannot be redistributed."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute embeddings even if a cached embedding file exists.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    start_time = time.time()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TRUST_REMOTE_CODE", "True")
    save_parameters(args, output_dir)

    log("Loading patent data...")
    df = read_table(input_path)
    documents = build_documents(df, args.title_col, args.abstract_col).tolist()
    log(f"Number of documents: {len(documents):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading embedding model on {device}...")
    embedding_model = SentenceTransformer(args.model_path, device=device)

    embedding_path = output_dir / "document_embeddings.npy"
    if embedding_path.exists() and not args.no_cache:
        log("Loading cached document embeddings...")
        embeddings = np.load(embedding_path)
    else:
        log("Computing document embeddings...")
        embeddings = embedding_model.encode(
            documents,
            show_progress_bar=True,
            batch_size=args.batch_size,
        )
        np.save(embedding_path, embeddings)

    final_stopwords = sorted(set(ENGLISH_STOP_WORDS).union(PATENT_STOPWORDS))
    log(f"Stopword list size: {len(final_stopwords)}")

    umap_model = UMAP(
        n_neighbors=args.umap_neighbors,
        n_components=args.umap_components,
        min_dist=0.0,
        metric="cosine",
        random_state=args.random_state,
    )

    cluster_model = KMeans(
        n_clusters=args.n_topics,
        n_init=10,
        random_state=args.random_state,
    )

    vectorizer_model = CountVectorizer(
        stop_words=final_stopwords,
        ngram_range=(1, 2),
        min_df=args.min_df,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        representation_model=MaximalMarginalRelevance(diversity=args.mmr_diversity),
        verbose=True,
    )

    log(f"Training topic model with K-Means (n_topics={args.n_topics})...")
    topics, _ = topic_model.fit_transform(documents, embeddings)
    topics_array = np.asarray(topics)

    log("Saving topic modeling outputs...")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_dir / "topic_info.csv", index=False)

    if args.id_col and args.id_col in df.columns:
        doc_ids: Iterable = df[args.id_col]
    else:
        doc_ids = range(len(df))

    doc_topic_df = pd.DataFrame({
        "Doc_ID": doc_ids,
        "Topic": topics_array,
    })

    if args.include_text_output:
        doc_topic_df[args.title_col] = df[args.title_col]
        doc_topic_df[args.abstract_col] = df[args.abstract_col]

    doc_topic_df.to_csv(output_dir / "documents_per_topic.csv", index=False)

    topic_model.save(output_dir / "bertopic_model", serialization="safetensors")

    noise_count = int(np.sum(topics_array == -1))
    elapsed = (time.time() - start_time) / 60
    log(f"Completed topic modeling in {elapsed:.2f} minutes.")
    log(f"Number of outlier documents labeled as Topic -1: {noise_count}")
    print(topic_info[["Topic", "Count", "Name", "Representation"]].head(10))


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
