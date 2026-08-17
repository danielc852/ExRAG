# ExRAG

ExRAG is a reproducible retrieval-augmented generation (RAG) baseline for EnterpriseRAG-Bench. It provides a local data preparation and FAISS retrieval pipeline, simple and deep agent options, and evaluation workflows for comparing results.

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
