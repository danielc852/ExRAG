# 1. ExRAG

ExRAG is an experimental retrieval-augmented generation (RAG) project built on
EnterpriseRAG-Bench. It provides a reproducible pipeline for downloading and
freezing benchmark data, chunking documents, creating embeddings, building a
FAISS vector index, retrieving relevant evidence, and evaluating generated
answers.

The project supports:

- A direct-model baseline with no retrieval.
- Simple and deep retrieval agents.
- Ollama or OpenRouter for answer generation.
- OpenRouter embeddings, with optional local MLX embeddings on Apple Silicon.
- Sample and full benchmark runs with resumable artifacts and evaluation
  outputs.

# 2. Setup and run

## Requirements

- Python 3.11 to 3.14.
- [uv](https://docs.astral.sh/uv/) for dependency and virtual-environment
  management.
- An `OPENROUTER_API_KEY` for the default embedding engine.
- [Ollama](https://ollama.com/) running locally for the default answer-generation
  provider. This is not required when OpenRouter is used for generation.
- Sufficient disk space for the benchmark data, model files, embeddings, and
  FAISS index.

Optional: an Apple Silicon Mac with macOS 14 or later for local MLX embeddings.

## Install

Clone the repository, enter the project directory, and install the dependencies:

```bash
git clone <repository-url>
cd EnterpriseRAG
uv sync
```

To include development tools such as pytest:

```bash
uv sync --extra dev
```

For optional MLX embedding support:

```bash
uv sync --extra mlx
```

## Configure

Set the OpenRouter API key used by the default embedding engine:

```bash
export OPENROUTER_API_KEY="<your-openrouter-api-key>"
```

For the default Ollama generation provider, start Ollama and download the
default model:

```bash
ollama serve
ollama pull hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
```

`ollama serve` can be omitted if Ollama is already running. The main optional
environment variables are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Authenticates OpenRouter embedding and generation requests | Required for OpenRouter |
| `OPENROUTER_BASE_URL` | Overrides the OpenRouter API endpoint | `https://openrouter.ai/api/v1` |
| `LLM_PROVIDER` | Selects `ollama` or `openrouter` for generation | `ollama` |
| `LLM_MODEL` | Selects the generation model | Ollama LFM2.5 model above |
| `OLLAMA_BASE_URL` | Sets the Ollama server address | `http://localhost:11434` |
| `EMBEDDING_ENGINE` | Selects `openrouter` or `mlx` | `openrouter` |
| `EMBEDDING_MODEL` | Selects the embedding model | `nvidia/nemotron-3-embed-1b:free` |
| `RAG_ARTIFACT_ROOT` | Sets the artifact directory | `artifacts` |

## Run the sample pipeline

The sample pipeline uses 1,000 documents and 10 questions. Run its three stages
in order:

```bash
uv run python main.py download sample
uv run python main.py init_vectordb sample
uv run python main.py run_exper sample
```

The commands download and freeze the sample dataset, create the embeddings and
FAISS index, and then run the evaluation experiment. Artifacts are stored in
`artifacts/sample`, and evaluation results are written to the default run output
directory shown by the command.

Use `full` instead of `sample` in all three commands to run the complete
benchmark:

```bash
uv run python main.py download full
uv run python main.py init_vectordb full
uv run python main.py run_exper full
```

## Common run options

Choose the answer method, generation provider, model, retrieval depth, or output
directory when running an experiment:

```bash
uv run python main.py run_exper sample \
  --agent simple \
  --llm ollama \
  --model hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M \
  --top-k 5 \
  --output-dir runs/sample/simple-ollama
```

Supported agent methods are `baseline`, `simple`, and `deep`. To use OpenRouter
for generation, set `--llm openrouter` and pass a compatible model name with
`--model`.

Run the automated tests with:

```bash
uv run pytest
```

Use the built-in help to see every available option:

```bash
uv run python main.py --help
uv run python main.py <command> --help
```
