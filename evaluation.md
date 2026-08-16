# EnterpriseRAG LangSmith Evaluation Suite

## 1. Purpose

呢個 repository 有兩條互補嘅 evaluation path：

1. `python main.py eval`：保留本地、可 resume 嘅 answer generation，同時輸出官方相容 `answers.jsonl`。
2. `python main.py langsmith ...`：將同一份 frozen benchmark questions 同 agent target 放入 LangSmith，保存 traces、deterministic feedback、summary metrics 同 experiment comparisons。

兩條 path 必須共用同一個 `AgentRunResult` contract、同一個 frozen question snapshot，同一個 FAISS index lineage。LangSmith 唔會重新下載 Hugging Face dataset，亦唔會將 gold answers、answer facts 或 expected document IDs傳入agent。

```text
artifacts/source/questions.parquet
              │
              ├── local eval ──────→ answers.jsonl + run_details.jsonl
              │
              └── LangSmith dataset
                         ↓
                  simple / deep target
                         ↓
               deterministic evaluators
                         ↓
                  cloud experiment
                         ↓
        answers.jsonl + records.jsonl + summary.json
```

## 2. Official Evaluation Boundary

EnterpriseRAG-Bench官方 leaderboard使用GPT-5.4 Medium Reasoning判斷answer correctness、answer completeness同Invalid Extra Documents。呢啲判斷涉及語意同candidate document relevance，唔可以用簡單set comparison冒充。

LangSmith第一版只提供可重現嘅code-based metrics：

- Document recall。
- Strict extra document count；只代表candidate IDs減gold IDs，唔等同官方Invalid Extra Docs。
- Retrieved document count。
- Tool-call count。
- Run success及answer presence。
- Latency。
- Input/output tokens；provider冇提供時為null。

正式benchmark仍然要將本repo產生嘅`answers.jsonl`交畀EnterpriseRAG-Bench官方evaluation harness。

## 3. Package Structure

Root-level `eval.py`已遷移成package，避免local同cloud logic混埋同一個module：

```text
eval/
├── __init__.py
├── models.py
├── local.py
├── dataset.py
├── evaluators.py
├── experiment.py
└── results.py
```

Responsibilities：

- `models.py`：Pydantic configs、sync result、experiment record、summary同comparison models。
- `local.py`：原有resumable local evaluation、official answers同local metrics。
- `dataset.py`：將frozen questions同步成immutable LangSmith dataset snapshot。
- `evaluators.py`：row-level同experiment-level deterministic evaluators。
- `experiment.py`：target factory、experiment metadata同`Client.evaluate()` wiring。
- `results.py`：LangSmith result normalization、本地artifacts同experiment comparison。
- `__init__.py`：只re-export穩定public API，保留現有`from eval import ...` imports。

## 4. LangSmith Dataset Contract

### Snapshot identity

CLI接受dataset base name，預設為`EnterpriseRAG-Bench`。實際LangSmith dataset name係：

```text
{base_name}-{source_output_fingerprint[:12]}
```

因此每個本地frozen source snapshot都有獨立cloud dataset。Dataset metadata保存完整fingerprint、dataset revision、question count同artifact schema version。

### Example schema

Inputs只包含target可以睇到嘅資料：

```json
{
  "question_id": "qst_0001",
  "question": "..."
}
```

Reference outputs只供evaluators使用：

```json
{
  "gold_answer": "...",
  "expected_doc_ids": ["dsid_..."],
  "answer_facts": ["..."]
}
```

Example metadata：

```json
{
  "question_type": "semantic",
  "source_types": ["slack"],
  "ordinal": 0,
  "source_fingerprint": "...",
  "schema_version": 2
}
```

每個example加入`test` split。Example UUID使用dataset snapshot name同question ID經UUIDv5產生，令sync可重複。Sync規則：

- Dataset不存在：建立dataset同全部examples。
- Dataset及全部examples一致：idempotent no-op。
- Partial sync：只補回缺少examples。
- 現有example內容、metadata或unexpected example IDs不一致：拒絕覆寫，要求使用新base name；第一版唔刪除cloud data。

## 5. Target and Evaluator Contracts

LangSmith target只接收example inputs，並返回：

```json
{
  "question_id": "qst_0001",
  "answer": "...",
  "document_ids": ["dsid_..."],
  "tool_calls": [],
  "latency_ms": 123.4,
  "input_tokens": 100,
  "output_tokens": 20,
  "model_name": "qwen3:8b",
  "agent_mode": "simple",
  "error": null
}
```

Nested LangChain／LangGraph calls由LangSmith tracing context捕捉。每題error會結構化寫入output，唔會中止其他examples。

Row evaluator一次返回以下feedback：

| Key | Value | Direction |
| --- | --- | --- |
| `document_recall` | 0–1；empty gold為null | higher |
| `strict_extra_documents` | integer；empty gold為null | lower |
| `retrieved_document_count` | integer | diagnostic |
| `tool_call_count` | integer | diagnostic |
| `run_success` | boolean | higher |
| `answer_present` | boolean | higher |
| `latency_ms` | float | lower |
| `input_tokens` | integer或null | lower |
| `output_tokens` | integer或null | lower |

Summary evaluator計算mean document recall、mean strict extras、failure rate、mean/p95 latency、mean tool calls同mean token usage。Null values唔會當零分。

## 6. CLI and Runtime Behavior

### Environment

```bash
export LANGSMITH_API_KEY="..."
export LANGSMITH_TRACING=true
# Optional:
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_WORKSPACE_ID="..."
```

LangSmith commands係cloud-required。API key缺少時要即時回傳可操作錯誤；credentials永遠唔寫入config、artifacts或Git。

### Dataset sync

```bash
python main.py langsmith sync \
  --artifact-root artifacts \
  --dataset-name EnterpriseRAG-Bench
```

### Run experiment

```bash
python main.py langsmith run \
  --agent simple \
  --artifact-root artifacts \
  --dataset-name EnterpriseRAG-Bench \
  --limit-questions 10 \
  --max-concurrency 1
```

`run`要求dataset已sync並同source fingerprint一致。預設每次建立新immutable experiment；第一版唔extend舊experiment。Experiment metadata保存Git commit、agent/model/top-k、question selection、dataset/index/embedding fingerprints、chunking config同corpus mode。

Default concurrency係1。提高concurrency時，retriever嘅SQLite metadata reads會使用thread-safe connection加lock；FAISS只執行read-only search。

### Compare experiments

```bash
python main.py langsmith compare \
  SIMPLE_EXPERIMENT_NAME \
  DEEP_EXPERIMENT_NAME
```

Comparison前必須驗證兩個experiments使用相同dataset/source/index/model/top-k同question selection。Agent mode可以不同。Report包含：

- 兩邊aggregate metrics。
- Raw `B - A` delta同higher/lower-is-better方向。
- B嘅document recall下降或success由true變false嘅question IDs。

Comparison唔會執行LLM pairwise judge。

## 7. Local Artifacts

每個LangSmith experiment完成後保存：

```text
runs/langsmith/<experiment-name>/
├── experiment.json
├── answers.jsonl
├── records.jsonl
└── summary.json
```

- `experiment.json`：experiment ID/name/URL、dataset ID/name同完整non-secret config。
- `answers.jsonl`：官方EnterpriseRAG格式，按question ordinal穩定排序。
- `records.jsonl`：run ID、inputs、outputs、feedback同error。
- `summary.json`：deterministic aggregates。

Comparison輸出：

```text
runs/langsmith/comparisons/<timestamp>-<A>-vs-<B>.json
```

所有JSON／JSONL先寫temporary file再atomic rename。

## 8. Privacy and Reproducibility

LangSmith cloud會收到benchmark questions、reference answers、answer facts、document IDs、agent outputs，同nested retrieval traces內嘅retrieved chunk content。EnterpriseRAG-Bench係synthetic public benchmark，因此本project接受呢個上載範圍；唔應該未經審核將相同設定用於真正企業敏感資料。

Experiment comparison只接受一致lineage，避免將sample/full corpus、不同embedding、不同chunking或不同question subsets混為同一比較。

## 9. Official References

- [LangSmith Evaluation Quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [Define a Code Evaluator](https://docs.langchain.com/langsmith/code-evaluator-sdk)
- [Define a Summary Evaluator](https://docs.langchain.com/langsmith/summary)
- [Manage and Version Datasets](https://docs.langchain.com/langsmith/manage-datasets)
- [Read Experiment Results Locally](https://docs.langchain.com/langsmith/read-local-experiment-results)
- [Trace LangChain Applications](https://docs.langchain.com/langsmith/trace-with-langchain)
- [EnterpriseRAG-Bench Evaluation Quickstart](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/quickstart.md)
