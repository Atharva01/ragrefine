"""Generate and evaluate reproducible frozen BEIR baseline artifacts."""

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.beir.config import DATASETS, RETRIEVER, DatasetConfig, baseline_config
from benchmarks.beir.metrics import evaluate_snapshot
from benchmarks.beir.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    load_snapshot,
    snapshot_checksum,
    write_snapshot,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _dataset_config(name: str) -> DatasetConfig:
    return next(dataset for dataset in DATASETS if dataset.name == name)


def generate(dataset: DatasetConfig, output_dir: Path) -> Path:
    """Download one dataset and persist the configured original Top-N ranking."""
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval import models
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("run: uv sync --extra beir") from error

    archive_url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
        + dataset.name
        + ".zip"
    )
    data_path = Path(util.download_and_unzip(archive_url, str(output_dir / "datasets")))
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(
        split="test"
    )
    model_path = snapshot_download(
        repo_id=RETRIEVER.model,
        revision=RETRIEVER.revision,
        allow_patterns=[
            "1_Pooling/config.json",
            "config.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
    )
    search = DRES(
        models.SentenceBERT(
            model_path,
            device=RETRIEVER.device,
            local_files_only=True,
        ),
        batch_size=128,
    )
    results = EvaluateRetrieval(search, score_function="cos_sim").retrieve(
        corpus, queries
    )
    frozen_queries: list[dict[str, Any]] = []
    for query_id in sorted(queries):
        ranking = sorted(
            results[query_id].items(), key=lambda item: (-item[1], item[0])
        )
        candidates: list[dict[str, object]] = []
        for rank, (document_id, score) in enumerate(
            ranking[: RETRIEVER.top_n], start=1
        ):
            document = corpus[document_id]
            text = "\n".join(
                filter(None, [document.get("title", ""), document["text"]])
            )
            candidates.append(
                {"id": document_id, "text": text, "rank": rank, "score": score}
            )
        frozen_queries.append(
            {"id": query_id, "text": queries[query_id], "candidates": candidates}
        )
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        **baseline_config(dataset, RETRIEVER),
        "queries": frozen_queries,
        "qrels": qrels,
    }
    path = output_dir / "snapshot.json"
    write_snapshot(snapshot, path)
    return path


def evaluate(snapshot_path: Path, output_dir: Path) -> None:
    """Write baseline metrics and machine-readable run metadata."""
    snapshot = load_snapshot(snapshot_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "config.json",
        {"dataset": snapshot["dataset"], "retriever": snapshot["retriever"]},
    )
    _write_json(output_dir / "metrics.json", evaluate_snapshot(snapshot))
    _write_json(
        output_dir / "environment.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
            "dataset": snapshot["dataset"],
            "retriever": snapshot["retriever"],
            "snapshot_checksum": snapshot_checksum(snapshot),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "evaluate", "reproduce"))
    parser.add_argument(
        "--dataset", choices=[item.name for item in DATASETS], default="scifact"
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/b0")
    )
    args = parser.parse_args()
    snapshot = (
        generate(_dataset_config(args.dataset), args.output_dir)
        if args.command == "generate"
        else args.snapshot
    )
    if snapshot is None:
        parser.error("--snapshot is required for evaluate and reproduce")
    first = evaluate_snapshot(load_snapshot(snapshot))
    if args.command == "reproduce" and first != evaluate_snapshot(
        load_snapshot(snapshot)
    ):
        raise RuntimeError("frozen snapshot metrics were not reproducible")
    evaluate(snapshot, args.output_dir)


if __name__ == "__main__":
    main()
