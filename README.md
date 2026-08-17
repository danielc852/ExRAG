# ExRAG

ExRAG is a reproducible retrieval-augmented generation (RAG) baseline for EnterpriseRAG-Bench. It provides a local data preparation and FAISS retrieval pipeline, simple and deep agent options, and evaluation workflows for comparing results.

## Quick start

The experiment CLI has three pipelines. Choose `sample` for the 1,000-document,
10-question smoke run, or `full` for the complete benchmark. The sample corpus always
includes the gold documents for those 10 questions, then uses the configured seed to
fill the remaining document slots:

```bash
python main.py download sample
python main.py init_vectordb sample
python main.py run_exper sample
```

Replace `sample` with `full` in all three commands for a full experiment. By
default, the two modes use separate `artifacts/sample` and `artifacts/full`
directories.

The reproducible record for the rebuilt 2026-08-17 sample run, including its
scope, models, input sources, output files, and per-question measurements, is in
[`sample_experiment_2026-08-17.md`](sample_experiment_2026-08-17.md).

## Evaluate different RAG methods and models

ExRAG supports controlled experiments across four main variables:

| Variable | CLI option | Supported choices |
| --- | --- | --- |
| Agent method | `--agent` | `simple` or `deep` |
| Generation model | `--model` | Any compatible chat model already installed in Ollama |
| Retrieval depth | `--top-k` | Default number of chunks requested by the retrieval tool; default `5`, maximum `20` |
| Embedding backend/model | `--embedding-engine`, `--embedding-model` | `sentence-transformers` or compatible `mlx` embeddings |

The `simple` method is a bounded LangChain tool-calling agent with at most three
retrieval calls. The `deep` method uses Deep Agents for multi-step retrieval and
planning, with at most eight retrieval calls and eight model calls. Both methods
use the same grounded system prompt, FAISS retriever, SQLite metadata store, and
structured result format.

### Compare agent methods

Use the same corpus, vector index, generation model, and top-k value. Give every
run a separate output directory and use `--no-resume` so results from an older
configuration cannot be mixed into the new experiment:

```bash
python main.py run_exper sample \
  --agent simple \
  --model lfm2.5-2.6b:q4_k_m \
  --top-k 5 \
  --output-dir runs/sample/simple-lfm-k5 \
  --no-resume

python main.py run_exper sample \
  --agent deep \
  --model lfm2.5-2.6b:q4_k_m \
  --top-k 5 \
  --output-dir runs/sample/deep-lfm-k5 \
  --no-resume
```

### Compare generation models

Pull each model into Ollama first, then keep the agent method, vector index, and
top-k fixed. Model tags must match a model returned by the local Ollama server:

```bash
ollama pull <model-a>
ollama pull <model-b>

python main.py run_exper sample \
  --agent simple \
  --model <model-a> \
  --top-k 5 \
  --output-dir runs/sample/simple-model-a-k5 \
  --no-resume

python main.py run_exper sample \
  --agent simple \
  --model <model-b> \
  --top-k 5 \
  --output-dir runs/sample/simple-model-b-k5 \
  --no-resume
```

Generation-model comparisons do not require rebuilding the vector store. The
same embedding model recorded in the index manifest is automatically loaded for
every retrieval query.

### Compare embedding models or chunking strategies

Embedding and chunking changes require a new vector index. Use a distinct
artifact root for every retrieval configuration so indexes and their lineage
remain isolated:

```bash
# Retrieval configuration A
python main.py download sample \
  --artifact-root artifacts/experiments/bge-base
python main.py init_vectordb sample \
  --artifact-root artifacts/experiments/bge-base \
  --embedding-engine sentence-transformers \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --chunk-size 512 \
  --chunk-overlap 64
python main.py run_exper sample \
  --artifact-root artifacts/experiments/bge-base \
  --agent simple \
  --model lfm2.5-2.6b:q4_k_m \
  --output-dir runs/sample/bge-base-simple \
  --no-resume

# Retrieval configuration B
python main.py download sample \
  --artifact-root artifacts/experiments/embedding-b
python main.py init_vectordb sample \
  --artifact-root artifacts/experiments/embedding-b \
  --embedding-engine sentence-transformers \
  --embedding-model <embedding-model-b> \
  --chunk-size 384 \
  --chunk-overlap 48
python main.py run_exper sample \
  --artifact-root artifacts/experiments/embedding-b \
  --agent simple \
  --model lfm2.5-2.6b:q4_k_m \
  --output-dir runs/sample/embedding-b-simple \
  --no-resume
```

For a fair retrieval comparison, keep the dataset revision, sample seed,
generation model, agent method, and top-k unchanged. Change one retrieval
variable at a time unless the experiment explicitly studies a combined system.

### Experiment outputs and metrics

Each local run writes the following files under its `--output-dir`:

| File | Purpose |
| --- | --- |
| `config.json` | Agent/model settings plus the complete dataset and index lineage |
| `answers.jsonl` | Final answers and retrieved document IDs in benchmark-compatible format |
| `run_details.jsonl` | Successful per-question tool traces, end-to-end latency, token counts, and recall |
| `summary.json` | Completion count, mean document recall, strict extras, latency, and tool calls |
| `errors.jsonl` | Created only when one or more questions fail |

`latency_ms` is the input-to-complete-output time for one question. It includes
all model turns, retrieval calls, and agent orchestration; it is not time to
first token or raw model tokens per second. Retrieval calls also record their
own latency inside `tool_calls`.

Recommended comparison dimensions are:

- Answer correctness and completeness through the official
  EnterpriseRAG-Bench evaluator.
- Document recall and extra retrieved documents.
- End-to-end latency and retrieval latency.
- Input/output token usage and tool-call count.
- Failure rate and missing-answer count.

The local deterministic metrics do not replace the official semantic evaluator.
For cloud traces, p95 latency, experiment-level summaries, and compatible-run
comparisons, use the optional LangSmith Python workflow documented in
[`evaluation.md`](evaluation.md).

Experiment outputs can contain benchmark questions, generated answers,
retrieved document IDs, and retrieved text in traces. Review or redact them
before sharing, especially when adapting this pipeline to non-synthetic data.

## MLX embeddings on Apple Silicon

The vector pipeline supports two embedding engines: the existing
`sentence-transformers` backend and an optional native `mlx-embeddings` backend.
Install the MLX extra with a native ARM Python on an Apple Silicon Mac running
macOS 14 or newer:

```bash
uv sync --extra mlx
```

Then build the sample vector store with an MLX-converted text embedding
checkpoint:

```bash
python main.py init_vectordb sample \
  --embedding-engine mlx \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rebuild
```

You can also set `EMBEDDING_ENGINE=mlx` and `EMBEDDING_MODEL=<model-or-path>`.
`mlx-embeddings` supports only specific architectures and expects compatible
MLX weights. To retain `BAAI/bge-base-en-v1.5` specifically, first convert it on
the Mac and pass the converted local directory:

```bash
python -m mlx_embeddings.convert \
  --hf-path BAAI/bge-base-en-v1.5 \
  --mlx-path models/bge-base-en-v1.5-mlx \
  --dtype bfloat16

python main.py init_vectordb sample \
  --embedding-engine mlx \
  --embedding-model models/bge-base-en-v1.5-mlx \
  --rebuild
```

The embedding manifest records the engine, model, and optional revision. Query
retrieval loads those exact settings, so document and query vectors cannot
silently use different backends. Embeddings remain normalized `float32` NumPy
shards and the FAISS cosine-similarity index format is unchanged. Changing the
engine or model requires `--rebuild`; existing Sentence Transformers indexes
remain readable because manifests without an engine field default to
`sentence-transformers`.
