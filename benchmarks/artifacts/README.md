# Retained benchmark artifacts

Benchmark inputs are immutable external artifacts, not repository source files.
The committed `retained-v1.json` manifest inventories every input needed to
reproduce retained B0/B1-reference/B2-L metrics and pins its SHA-256.

The bundle must be extracted so its paths are rooted at `benchmarks/`; for
example, `benchmarks/results/scifact-b0/snapshot.json` must exist. The complete
bundle is intentionally external because the frozen JSON snapshots and rankings
are large. It must include each listed JSON file and every listed snapshot or
ranking `.sha256` sidecar.

From a clean checkout, obtain the immutable `retained-v1` artifact bundle from
the project artifact store, extract it into the repository's `benchmarks/`
directory, then verify and reproduce without retrieval or model inference:

```powershell
uv run python -m benchmarks.beir.artifacts verify `
  --manifest benchmarks/artifacts/retained-v1.json `
  --artifact-root benchmarks

uv run python -m benchmarks.beir.cross_dataset run `
  --output-dir benchmarks/results/cross-dataset-reproduced-v1

uv run python -m benchmarks.beir.cross_dataset reproduce `
  --output-dir benchmarks/results/cross-dataset-reproduced-v1
```

`verify` fails with the missing path or mismatched SHA-256 before an analysis
runner starts. The manifest is a distribution contract, not a result artifact;
new benchmark runs must use new result directories and a new manifest version.

## Inventory policy

| Artifact class | Repository policy | Reproduction role |
| --- | --- | --- |
| Source, manifests, scripts, and small human-readable reports | Committed | Defines the experiment and verification procedure. |
| B0 snapshots, B1/B2 persisted rankings, environment files, and their sidecars | External `retained-v1` bundle | Required to recompute retained SciFact/NFCorpus/FiQA metrics. |
| Generated analysis directories | Local/generated | May be recreated from the verified bundle; never overwrite an existing directory. |
| B3/B4 historical result directories | External historical evidence | Not required for the retained B0/B1-reference/B2-L quality replay; their tracked report pointers remain references only. |
