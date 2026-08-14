# EnterpriseRAG LangSmith Evaluation Pipeline

> 呢份文件定義 EnterpriseRAG 嘅 evaluation code structure。目標係用同一套 benchmark 同 metrics，比較 v1 exact vector search、v2 HNSW，以及之後嘅 system versions。

## 1. 設計目標

Evaluation code 應該按責任拆分，而唔係按 LangSmith API function 拆分。第一版只保留四個主要 responsibility files；`__init__.py` 只係 Python package marker，唔放 business logic：

```text
eval/
├── __init__.py
├── data.py
├── eval.py
├── results.py
└── build_exp.py
```

四個 files 分別回答四個問題：

```text
data.py      = What do we test?
eval.py      = How do we evaluate?
results.py   = What did we learn?
build_exp.py = How do we run the whole experiment?
```

LangSmith evaluation 嘅基本流程係 dataset → target function → evaluators → experiment → results。每次 version comparison 都應該沿用同一條 evaluation pipeline。

## 2. 整體 Pipeline

```text
                         build_exp.py
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
            data.py        system target    eval.py
               │           v1 / v2 / ...      │
               │              │               │
               └──────────────┼───────────────┘
                              ▼
                    LangSmith experiment
                              │
                              ▼
                         results.py
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
              metrics.json          report.html
```

Execution flow：

1. `build_exp.py` 接收 version 同 experiment configuration。
2. `data.py` 取得固定 benchmark dataset。
3. System layer 建立該 version 嘅 target application。
4. `eval.py` 用 LangSmith 執行 target 同 evaluators。
5. `results.py` 將 experiment results 正規化、統計及可視化。
6. Artifacts 保存到對應 version folder，供比較同重建報告。

## 3. `data.py` — Dataset Layer

`data.py` 只負責 benchmark data：讀取、驗證同正規化 dataset，然後向其他 modules 提供穩定介面。

### 應負責

- 由 LangSmith dataset、local JSON 或其他指定來源讀取 benchmark。
- 驗證每個 example 都有 question、reference outputs 同穩定 example ID。
- 將唔同來源轉成相同 schema。
- 固定 dataset name／ID、revision、split 同 question manifest。
- 如有需要，將 local benchmark 上載或同步到 LangSmith dataset。

### 唔應負責

- Flat 或 HNSW index implementation。
- Retrieval、LLM call 或 agent orchestration。
- Recall calculation。
- Experiment execution。
- Visualization。

### 建議介面

```python
# eval/data.py

from langsmith import Client


def load_dataset(client: Client, dataset_name: str):
    """Return the fixed LangSmith dataset used by every system version."""
    return client.read_dataset(dataset_name=dataset_name)


def normalize_example(row: dict) -> dict:
    """Convert a benchmark row into the shared evaluation schema."""
    return {
        "id": row["question_id"],
        "inputs": {
            "question": row["question"],
        },
        "reference_outputs": {
            "relevant_doc_ids": row["expected_doc_ids"],
        },
    }
```

其他 modules 唔需要知道 dataset 係來自 LangSmith、JSON 定其他 storage，只需要取得同一種 normalized dataset handle 或 examples。

## 4. `eval.py` — Evaluation Engine

`eval.py` 係 evaluation 核心，負責定義 evaluators，同埋呼叫 LangSmith 執行 benchmark。佢接收已建立嘅 target，但唔應該自己選擇 Flat 或 HNSW。

### 應負責

- 建立 code evaluators。
- 呼叫 `Client.evaluate()`。
- 設定 dataset、experiment name、metadata、concurrency 同 blocking mode。
- 返回 `ExperimentResults`，交畀 `results.py` 處理。
- 將 evaluator 定義保持跨 versions 一致。

### v1 最小 evaluator

第一個 deterministic metric 係 Recall@5：

```python
# eval/eval.py

from langsmith import Client


def recall_at_5(outputs: dict, reference_outputs: dict) -> float:
    expected = set(reference_outputs["relevant_doc_ids"])
    retrieved = set(outputs["retrieved_doc_ids"][:5])

    if not expected:
        raise ValueError("relevant_doc_ids must not be empty")

    return len(expected & retrieved) / len(expected)


def build_evaluators():
    return [recall_at_5]
```

例如 gold documents 係 `A, B, C`，retrieved top 5 係 `A, D, B, E, F`，Recall@5 就係 `2 / 3 = 0.667`。

### 執行 experiment

```python
def run_evaluation(
    client: Client,
    *,
    target,
    dataset_name: str,
    experiment_name: str,
    metadata: dict,
):
    return client.evaluate(
        target,
        data=dataset_name,
        evaluators=build_evaluators(),
        experiment_prefix=experiment_name,
        metadata=metadata,
        blocking=True,
    )
```

`target` 必須接收 dataset example 嘅 `inputs` dictionary，並返回標準 output dictionary。LangSmith 會逐個 example 執行 target、將結果交畀 evaluators，再將 outputs、scores 同 traces 保存為 experiment。

日後可以加入 answer correctness、groundedness、answer relevance 或 pairwise evaluation，但 v1 應先維持最少 metrics，確保 retrieval comparison 清晰。

## 5. `results.py` — Results and Visualization

`results.py` 唔會執行 RAG application，亦唔會重新評分。佢只處理 LangSmith `ExperimentResults` 或已保存嘅 normalized artifacts。

### 應負責

```text
ExperimentResults
        ↓
parse
        ↓
normalize
        ↓
aggregate
        ↓
compare
        ↓
visualize
        ↓
save artifacts
```

### 建議介面

```python
# eval/results.py

def results_to_records(results) -> list[dict]:
    records = []

    for item in results:
        run = item["run"]
        evaluation_results = item["evaluation_results"]["results"]

        records.append(
            {
                "run_id": str(run.id),
                "inputs": run.inputs,
                "outputs": run.outputs,
                "evaluation_results": evaluation_results,
            }
        )

    return records


def summarize_results(records: list[dict]) -> dict:
    """Return aggregate retrieval, latency, token and error metrics."""
    ...


def render_report(records: list[dict], summary: dict, output_path) -> None:
    """Generate a self-contained HTML report."""
    ...


def save_metrics(summary: dict, output_path) -> None:
    """Save machine-readable metrics.json."""
    ...
```

### 第一版報告

- Mean Recall@5。
- Retrieval latency p50、p95、p99。
- End-to-end latency。
- Token usage 同 cost，如 trace 有提供。
- Error rate 同 failure cases。
- 每題 retrieved documents、scores 同 evaluator result。
- Config、dataset、system version、index type 同 code commit。

`results.py` 可以理解成：

```text
LangSmith experiment → human-readable and machine-readable information
```

## 6. `build_exp.py` — Pipeline Orchestrator

`build_exp.py` 只負責連接 components，唔應該包含 dataset parsing、metric calculation、HNSW implementation 或 plotting details。

### 應負責

- 解析 CLI arguments。
- 讀取 version-specific config。
- 取得 dataset。
- 建立 system target。
- 呼叫 `run_evaluation()`。
- 將結果交畀 `results.py`。
- 建立 version artifact folder。

### 建議結構

```python
# eval/build_exp.py

from langsmith import Client

from eval.data import load_dataset
from eval.eval import run_evaluation
from eval.results import (
    render_report,
    results_to_records,
    save_metrics,
    summarize_results,
)
from pipeline import build_pipeline


def build_target(version: str):
    pipeline = build_pipeline(version=version)

    def target(inputs: dict) -> dict:
        return pipeline.invoke(inputs["question"])

    return target


def build_experiment(config: dict) -> dict:
    client = Client()
    dataset = load_dataset(client, config["dataset_name"])
    target = build_target(config["version"])

    results = run_evaluation(
        client,
        target=target,
        dataset_name=dataset.name,
        experiment_name=f'{config["version"]}-rag',
        metadata=config,
    )

    records = results_to_records(results)
    summary = summarize_results(records)
    save_metrics(summary, config["metrics_path"])
    render_report(records, summary, config["report_path"])
    return summary
```

上面係 responsibility skeleton，實作時應使用 typed config 同明確 error handling，唔應該直接用未驗證 dictionary paths。

## 7. Target Contract

Evaluation layer 要求所有 system versions 使用同一個介面。

### Input

```python
{
    "question": "What is the company's retention policy?"
}
```

### Reference output

```python
{
    "relevant_doc_ids": ["doc_12", "doc_18"]
}
```

### Target output

```python
{
    "answer": "...",
    "retrieved_doc_ids": [
        "doc_12",
        "doc_5",
        "doc_18",
        "doc_42",
        "doc_7",
    ],
    "retrieved_chunks": [
        {
            "chunk_id": "doc_12::3",
            "document_id": "doc_12",
            "rank": 1,
            "score": 0.87,
        }
    ],
    "retrieval_latency_ms": 42.0,
}
```

Evaluator 唔需要知道 target 底層用 Flat、HNSW、reranker 定 deep agent。只要所有 versions 返回相同 schema，evaluation code 就唔需要因 system implementation 改動。

## 8. v1 and v2 Comparison

```text
v1
FAISS exact flat search
        │
        │ only change index algorithm
        ▼
v2
HNSW
M=32
efConstruction=200
efSearch=64
```

以下項目必須保持一致：

- Dataset、split 同 question manifest。
- Chunk content、chunk IDs、chunk size 同 overlap。
- Embedding model、revision 同 normalized vectors。
- Cosine／inner-product scoring semantics。
- `top_k=5`。
- Agent、prompt、generation model 同 temperature。
- Evaluators 同 report calculations。
- Hardware、concurrency 同量度方法。

HNSW implementation 屬於 system／retrieval layer，唔屬於 `eval/`。建議 system structure：

```text
retrieval/
├── flat.py
└── hnsw.py

pipeline.py

eval/
├── data.py
├── eval.py
├── results.py
└── build_exp.py
```

## 9. Artifacts

```text
artifacts/
├── v1/
│   ├── config.json
│   ├── records.jsonl
│   ├── metrics.json
│   └── report.html
└── v2/
    ├── config.json
    ├── records.jsonl
    ├── metrics.json
    └── report.html
```

Visualization layer 應該只依賴 normalized artifacts，唔應該直接依賴 live LangSmith objects。咁舊 experiment 可以離線重建報告，亦方便日後建立跨版本 dashboard。

建議命令：

```bash
python -m eval.build_exp --version v1
python -m eval.build_exp --version v2
python -m eval.build_exp --compare v1 v2
```

Comparison report 最少包括：

```text
Recall@5
Retrieval latency p50 / p95 / p99
Index build time
Index size and memory
End-to-end latency
Token usage and cost
Failures and regression cases
```

## 10. 最小實作次序

- [ ] 建立 `eval/` folder 同四個 files。
- [ ] 在 `data.py` 接駁固定 benchmark dataset。
- [ ] 定義共用 target input／output schema。
- [ ] 在 `eval.py` 實作 Recall@5 同 LangSmith experiment execution。
- [ ] 在 `results.py` 將 `ExperimentResults` 轉成 normalized records。
- [ ] 生成 `metrics.json` 同最簡 `report.html`。
- [ ] 在 `build_exp.py` 接通 v1 end-to-end flow。
- [ ] 用完全相同 evaluation config 跑 v2 HNSW。
- [ ] 生成 v1 vs v2 comparison report。

第一版暫時唔需要拆出 `evaluators/`、`metrics/`、`plots/`、`experiments/` 或 `configs/`。等四個 files 任何一個責任明顯過大，先按實際需要拆分。

### 現有 repo migration note

目前 repo 已有 root-level `eval.py`。正式建立 `eval/` package 時，應將相關 evaluation logic 搬到 `eval/eval.py` 或其他對應 responsibility file，更新 imports 同 `pyproject.toml` 後先移除 root-level file。唔應該長期同時保留 root `eval.py` 同 `eval/` package，否則 `python -m eval.build_exp` 可能出現 import ambiguity。

## 11. LangSmith 官方參考

- [Evaluation overview](https://docs.langchain.com/langsmith/evaluation)
- [Evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [Evaluate an LLM application](https://docs.langchain.com/langsmith/evaluate-llm-application)
- [Define a code evaluator](https://docs.langchain.com/langsmith/code-evaluator-sdk)
- [Read experiment results locally](https://docs.langchain.com/langsmith/read-local-experiment-results)
