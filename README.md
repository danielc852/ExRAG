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
