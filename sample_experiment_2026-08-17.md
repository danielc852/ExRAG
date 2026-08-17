# Sample Experiment Record — 2026-08-17

## 1. 實驗目的

今次實驗驗證 sample RAG pipeline 由使用者輸入問題、FAISS retrieval、agent tool calls，到本機 LLM 產生完整答案嘅端到端表現。實驗亦確認重建後嘅 document chunks、embeddings、FAISS IDs 同 SQLite metadata 保持一致。

呢次係 sample smoke experiment，唔係完整 500-question benchmark，因此結果只適合驗證 pipeline 正常運作同作本機效能基線，唔應視為官方 benchmark 分數。

## 2. 測試範圍

| 項目 | 設定 |
| --- | --- |
| 執行日期 | 2026-08-17（Asia/Hong_Kong） |
| Git commit | `9475067a213d565989fd0992b64df1f7c24da51e` |
| Corpus mode | `sample` |
| Agent mode | `simple` |
| 問題數量 | 10 |
| 問題類型 | 10 × `basic` |
| Document 數量 | 1,000 |
| Chunk 數量 | 3,656 |
| Chunk size／overlap | 512／64 tokens |
| Retrieval index | `faiss.IndexIDMap2(IndexFlatIP)` |
| Similarity | Cosine similarity（normalized inner product） |
| Default top-k | 5；agent 可以喺個別 tool call 要求最多 20 |
| Experiment concurrency | Sequential local execution |

重建命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python main.py init_vectordb sample --rebuild
```

Experiment 命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python main.py run_exper sample \
  --model lfm2.5-2.6b:q4_k_m \
  --output-dir runs/sample/rebuild-20260817-simple \
  --no-resume
```

## 3. 模型

| 用途 | Engine／模型 | 設定 |
| --- | --- | --- |
| Document/query embedding | `sentence-transformers`／`BAAI/bge-base-en-v1.5` | 768 dimensions、`float32`、normalized、batch size 32 |
| Answer generation | Ollama／`lfm2.5-2.6b:q4_k_m` | Local inference、simple RAG agent |

Document 同 query embeddings 使用相同 engine 同模型，避免向量空間不一致。重建後 manifest 已保存 `embedding_engine: sentence-transformers`。

## 4. Input sources

原始資料來自 Hugging Face dataset `onyx-dot-app/EnterpriseRAG-Bench`，revision 為 `main`：

- Documents：`artifacts/sample/source/documents/part-000000.parquet`
- Frozen questions：`artifacts/sample/source/questions.parquet`
- Dataset document fingerprint：`1a7cc070927ca234`
- Dataset question fingerprint：`f23690ae0a89166f`
- Sampling seed：42
- Sample corpus 特別包含首 10 題需要嘅全部 gold documents，再隨機補足至 1,000 documents。

1,000 份 sample documents 嘅 source-type 分佈：

| Source type | Documents |
| --- | ---: |
| Slack | 551 |
| Gmail | 250 |
| Linear | 73 |
| Google Drive | 48 |
| HubSpot | 25 |
| GitHub | 16 |
| Fireflies | 15 |
| Confluence | 13 |
| Jira | 9 |

10 條測試問題所標示嘅來源包括 GitHub 3 條、Google Drive 3 條、Gmail 2 條、Fireflies 1 條同 Linear 1 條。Gold answers、answer facts 同 expected document IDs 只用於 evaluation，唔會傳入 agent。

## 5. Output files

Experiment output directory：`runs/sample/rebuild-20260817-simple/`

| File | 內容 |
| --- | --- |
| `config.json` | 執行設定、index manifest、dataset/model fingerprints |
| `answers.jsonl` | 每題最終答案同 retrieved document IDs；10 rows |
| `run_details.jsonl` | 每題完整 metrics、tool-call traces、token counts、latency；10 rows |
| `summary.json` | 全 experiment aggregate metrics |

因為 10 題全部成功，所以今次冇產生 `errors.jsonl`。

## 6. Latency 定義

`run_details.jsonl` 入面嘅 `latency_ms` 係 input-to-complete-output end-to-end latency：

```text
輸入問題
→ agent／LLM processing
→ vector retrieval tool call(s)
→ 後續 LLM turn(s)
→ 完整答案
```

佢包含 prompt processing、所有 LLM turns、retrieval 同 orchestration overhead。佢唔係 time to first token（TTFT），亦唔係純模型 tokens-per-second。每個 tool trace 嘅 `latency_ms` 就只量度該次 retrieval。

## 7. Recorded experiment summary

以下數據直接來自 `summary.json` 或由 `run_details.jsonl` 記錄計算：

| Metric | Result |
| --- | ---: |
| Attempted | 10 |
| Completed | 10 |
| Failed | 0 |
| Mean document recall | 1.0（100%） |
| Mean strict extra documents | 4.5 |
| Mean end-to-end latency | 23,949.10 ms（23.95 s） |
| Minimum end-to-end latency | 14,172.04 ms（14.17 s） |
| Maximum end-to-end latency | 38,985.86 ms（38.99 s） |
| Mean tool calls per question | 1.5 |
| Total input tokens | 101,566 |
| Total output tokens | 11,089 |
| Benchmark comparable | No；sample run only |

`strict_extra_documents` 只係 retrieved IDs 減去 gold IDs 嘅集合差異，唔等同官方以 LLM 判斷嘅 Invalid Extra Documents metric。

## 8. Per-question recorded data

`Tool latency` 係該題所有 retrieval calls 嘅 recorded latency 總和。其餘數值直接取自每題 run detail。

| Question | End-to-end latency (ms) | Tool calls | Tool latency (ms) | Input tokens | Output tokens | Document recall | Strict extras |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qst_0001` | 14,172.04 | 1 | 151.11 | 4,508 | 573 | 1.00 | 3 |
| `qst_0002` | 14,419.32 | 1 | 45.24 | 5,181 | 639 | 1.00 | 4 |
| `qst_0003` | 38,716.41 | 2 | 75.28 | 13,129 | 2,079 | 1.00 | 7 |
| `qst_0004` | 19,501.51 | 1 | 41.60 | 6,268 | 904 | 1.00 | 1 |
| `qst_0005` | 35,391.68 | 2 | 74.93 | 16,046 | 1,607 | 1.00 | 5 |
| `qst_0006` | 15,155.47 | 1 | 41.52 | 5,684 | 650 | 1.00 | 2 |
| `qst_0007` | 19,355.79 | 1 | 45.01 | 5,492 | 959 | 1.00 | 6 |
| `qst_0008` | 38,985.86 | 3 | 107.95 | 28,831 | 1,511 | 1.00 | 6 |
| `qst_0009` | 19,584.36 | 1 | 38.28 | 5,390 | 981 | 1.00 | 3 |
| `qst_0010` | 24,208.54 | 2 | 76.81 | 11,037 | 1,186 | 1.00 | 8 |

## 9. Post-run integrity checks

- FAISS vectors：3,656。
- SQLite chunks：3,656，覆蓋全部 1,000 documents。
- Embedding array：`(3656, 768)`、normalized `float32`。
- Parquet、NumPy、FAISS、SQLite counts 一致。
- 所有 experiment retrieved document IDs 都存在於重建後嘅 SQLite store。
- 10 個 expected gold documents 全部存在於 sample corpus 同 final index。
