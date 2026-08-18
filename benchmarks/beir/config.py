"""Versioned configuration for the frozen BEIR baseline."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """One BEIR dataset selected for the baseline experiment."""

    name: str
    version: str = "beir-2.2.0"


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Pinned first-stage dense retriever settings."""

    model: str = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
    revision: str = "14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3"
    backend: str = "torch"
    device: str = "cpu"
    top_n: int = 50


DATASETS = (
    DatasetConfig("scifact"),
    DatasetConfig("nfcorpus"),
    DatasetConfig("fiqa"),
)
RETRIEVER = RetrieverConfig()


def baseline_config(
    dataset: DatasetConfig, retriever: RetrieverConfig
) -> dict[str, object]:
    """Return the serializable configuration recorded with each run."""
    return {"dataset": asdict(dataset), "retriever": asdict(retriever)}
