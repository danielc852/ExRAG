# ExRAG

ExRAG is an experimental workbench for learning how to build a robust
retrieval-augmented generation (RAG) system with EnterpriseRAG-Bench. A
reproducible baseline is the starting point, not the final goal: each experiment
changes a controlled part of the pipeline, records its behavior, and uses the
evidence to guide the next iteration.

It provides a local data-preparation and FAISS retrieval pipeline, simple and
deep agent methods, interchangeable generation and embedding models, and local
or LangSmith evaluation workflows for comparing results.

## Project goal and learning loop

The project studies RAG as a complete system. A robust result should retrieve
the right evidence, answer faithfully, fail clearly when evidence is missing,
remain reproducible across runs, and expose enough latency and trace data to
diagnose regressions.

Progress is developed through the following loop:

```text
Establish a reproducible baseline
              ↓
Choose one retrieval, model, or agent hypothesis
              ↓
Run the same frozen evaluation scope
              ↓
Measure quality, retrieval, latency, cost, and failures
              ↓
Inspect per-question traces and failure cases
              ↓
Keep, revise, or reject the change
              ↓
Record the evidence and start the next experiment
```

The experiment programme covers these layers:

| Layer | What the experiments are intended to learn |
| --- | --- |
| Data integrity | Whether source documents, chunks, vectors, metadata, and questions share a verified lineage |
| Retrieval | How embedding models, chunking, overlap, and retrieval depth affect relevant-document recall and noise |
| Agent behavior | Whether simple or deeper multi-step retrieval improves grounded answers without unnecessary calls |
| Generation | How local LLM choice affects correctness, completeness, latency, token use, and instruction following |
| Robustness | How the system behaves with missing evidence, conflicting sources, retrieval misses, interrupted runs, or corrupted artifacts |
| Evaluation | Which local deterministic signals predict official semantic answer quality and which require an external judge |
| Operations and privacy | Whether runs are resumable, observable, reproducible, and safe to inspect or share |

Every meaningful experiment should state its hypothesis, frozen test scope,
changed variable, controlled variables, models, input lineage, output directory,
recorded metrics, observed failures, and conclusion. Sample runs provide a fast
learning loop; full runs determine whether an improvement generalizes across the
complete benchmark.

## Working plan

Current position: step 1 is implemented, the sample/test evaluation split is in
place, and the next milestone is to complete step 2 with a frozen 10-question
sample baseline. A cloud Ollama path has also been proven through an SSH tunnel;
it still needs to run the same sample evaluation before it can be compared with
the local baseline.

1. **Build the basic engine — complete.** Implement the end-to-end ingestion, chunking,
   embedding, indexing, retrieval, generation, and source-tracking pipeline.
2. **Establish the simple RAG baseline — current.** Freeze the 10-question
   `sample` scope and record
   answer quality, retrieval quality, latency, token usage, cost, and failures.
3. **Research algorithms and model options — next.** Compare relevant chunking,
   embedding, retrieval, reranking, generation-model, and agent approaches, then
   prioritize testable hypotheses.
4. **Optimize the system.** Improve one controlled variable at a time through
   context engineering, model selection, and retrieval-algorithm changes.
5. **Evaluate against the baseline.** Reuse the same frozen evaluation scope and
   compare both aggregate metrics and per-question failure cases with the
   previous accepted configuration.
6. **Iterate steps 4 and 5.** Keep, revise, or reject each change based on the
   recorded evidence, and promote a new configuration only when it provides a
   meaningful improvement within the project's quality, latency, and cost
   constraints.

Create the evaluation set before substantial optimization. ExRAG treats the
10-question `sample` dataset as the development/smoke set for repeated
experiments and the complete `test` dataset as held-out confirmation. Dataset
type is derived from the frozen artifact lineage, so a sample index cannot be
silently evaluated against held-out questions.

## Current code layout

The main runtime boundaries are deliberately small so experiments can change
one layer without coupling it to the others:

```text
agent/                         simple and deep RAG agent construction
data/                          frozen artifacts, processing, embeddings, and FAISS
providers/                     Ollama and OpenRouter generation providers
tools/                         retrieval tools exposed to the agents
eval/
├── dataset/                   frozen evaluation selection and LangSmith sync
├── runner.py                  resumable local evaluation and output writing
├── evaluators.py              deterministic row and summary metrics
├── experiment.py              LangSmith target and experiment orchestration
├── models.py                  evaluation configuration and result contracts
└── results.py                 result normalization and experiment comparison
```

`eval/dataset/` is the only subpackage under `eval`; use imports such as
`eval.dataset.sync`. The stable public evaluation API remains available from
`eval`, while local execution lives in `eval/runner.py`.

Generation is selected through the shared `providers/` interface. Ollama
remains the default, and OpenRouter can be selected with `--llm openrouter`.
Embedding is independent of generation: OpenRouter is the default embedding
engine, with MLX available as the optional local Apple Silicon backend. The
former Sentence Transformers engine is no longer supported, so indexes created
with it must be rebuilt.

## Quick start

The experiment CLI has three pipelines. Choose `sample` for the 1,000-document,
10-question smoke run, or `full` for the complete benchmark. The sample corpus always
includes the gold documents for those 10 questions, then uses the configured seed to
fill the remaining document slots:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python main.py download sample
python main.py init_vectordb sample
python main.py run_exper sample
```

The default embedding engine is OpenRouter with
`nvidia/nemotron-3-embed-1b:free`. Document chunks are sent in batches using
the `document` input type; retrieval questions use the `query` input type. The
API key is read only from `OPENROUTER_API_KEY` and is never stored in artifact
manifests.

Replace `sample` with `full` in all three commands for a full experiment. By
default, the two modes use separate `artifacts/sample` and `artifacts/full`
directories.

The reproducible record for the rebuilt 2026-08-17 sample run, including its
scope, models, input sources, output files, and per-question measurements, is in
[`sample_experiment_2026-08-17.md`](sample_experiment_2026-08-17.md).

## Evaluate different RAG methods and models

ExRAG supports controlled experiments across five main variables:

| Variable | CLI option | Supported choices |
| --- | --- | --- |
| Agent method | `--agent` | `simple` or `deep` |
| LLM provider | `--llm` | `ollama` (default) or `openrouter` |
| Generation model | `--model` | A concrete model name supported by the selected LLM provider |
| Retrieval depth | `--top-k` | Default number of chunks requested by the retrieval tool; default `5`, maximum `20` |
| Embedding engine/model | `--embedding-engine`, `--embedding-model` | `openrouter` or compatible `mlx` embeddings |

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

### Use OpenRouter for generation

Ollama remains the default generation provider, so existing commands continue
to work unchanged. To use a hosted OpenRouter chat model, select it explicitly;
`OPENROUTER_API_KEY` and optional `OPENROUTER_BASE_URL` are shared with the
embedding integration:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python main.py run_exper sample \
  --llm openrouter \
  --model openai/gpt-4.1-mini \
  --output-dir runs/sample/openrouter-gpt-4.1-mini \
  --no-resume
```

Chat and embedding requests use separate endpoint adapters while sharing the
same API key, base URL, timeout, and common OpenRouter headers. The selected LLM
provider and concrete model are both saved in run configuration metadata, so a
resumed run cannot silently switch either setting.

### Compare embedding models or chunking strategies

Embedding and chunking changes require a new vector index. Use a distinct
artifact root for every retrieval configuration so indexes and their lineage
remain isolated:

```bash
# Retrieval configuration A: hosted OpenRouter embeddings
python main.py download sample \
  --artifact-root artifacts/experiments/nemotron-openrouter
python main.py init_vectordb sample \
  --artifact-root artifacts/experiments/nemotron-openrouter \
  --embedding-engine openrouter \
  --embedding-model nvidia/nemotron-3-embed-1b:free \
  --tokenizer-model BAAI/bge-base-en-v1.5 \
  --chunk-size 512 \
  --chunk-overlap 64
python main.py run_exper sample \
  --artifact-root artifacts/experiments/nemotron-openrouter \
  --agent simple \
  --model lfm2.5-2.6b:q4_k_m \
  --output-dir runs/sample/nemotron-openrouter-simple \
  --no-resume

# Retrieval configuration B: local MLX embeddings on Apple Silicon
python main.py download sample \
  --artifact-root artifacts/experiments/bge-mlx
python main.py init_vectordb sample \
  --artifact-root artifacts/experiments/bge-mlx \
  --embedding-engine mlx \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --chunk-size 384 \
  --chunk-overlap 48
python main.py run_exper sample \
  --artifact-root artifacts/experiments/bge-mlx \
  --agent simple \
  --model lfm2.5-2.6b:q4_k_m \
  --output-dir runs/sample/bge-mlx-simple \
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

## OpenRouter embeddings

OpenRouter is the default embedding engine and requires no local Torch runtime.
Set an API key, then rebuild the vector artifacts because embeddings from a
different model or engine cannot share one FAISS index:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python main.py init_vectordb sample --rebuild
```

The defaults can be overridden explicitly:

```bash
python main.py init_vectordb sample \
  --embedding-engine openrouter \
  --embedding-model nvidia/nemotron-3-embed-1b:free \
  --tokenizer-model BAAI/bge-base-en-v1.5 \
  --rebuild
```

`OPENROUTER_BASE_URL` can override the default
`https://openrouter.ai/api/v1` endpoint. The tokenizer is configured separately
because an OpenRouter model slug is an API identifier, not necessarily a local
Hugging Face tokenizer identifier. Responses are validated, converted to
normalized `float32` arrays, and written to the existing sharded artifact
format.

## MLX embeddings on Apple Silicon

The vector pipeline also supports an optional native `mlx-embeddings` engine.
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
engine or model requires `--rebuild`. Indexes created with the removed Sentence
Transformers engine, including older manifests without an engine field, must be
rebuilt with OpenRouter or MLX before retrieval.
