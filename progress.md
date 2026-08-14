# EnterpriseRAG Progress and Version Plan

> 呢份文件係 `baseline.md` 嘅執行版：`baseline.md` 定義目標同固定規則；本文件追蹤目前狀態、下一步、每輪實驗，以及可視化輸出。

## 1. 核心原則

先完成一條最短而可重現嘅 end-to-end pipeline，再逐次增加複雜度。每個 version 只改一個主要變數，否則無法判斷結果改善或退步係由邊項改動造成。

首個可接受 baseline 暫時只用 `simple` agent；`deep` agent、reranker、hybrid search、複雜 dashboard 等功能留待 baseline 穩定後先比較。

## 2. 最簡 Baseline（v1）

### 2.1 結構

```text
EnterpriseRAG-Bench documents
        ↓
normalize → chunk (512/64)
        ↓
embedding → FAISS index
        ↓
question → retrieve top 5 chunks
        ↓
simple agent → grounded answer
        ↓
JSONL logs → summary.json → report.html
```

v1 唔需要額外 planner、query rewriting、reranker 或 metadata routing。目標係證明以下三件事：

1. 同一份 config 可以重建同一個 index。
2. 固定問題集可以由頭到尾完成，並留下完整 retrieval trace。
3. 每次 run 都可以生成可比較嘅 quality、latency 同 failure 報告。

### 2.2 固定設定

| 項目 | v1 設定 |
| --- | --- |
| Corpus | 固定 revision；開發期先用固定 1,000 documents |
| Chunking | 512 tokens，64 tokens overlap |
| Embedding | `BAAI/bge-base-en-v1.5`，固定 revision |
| Index | FAISS，cosine similarity |
| Retrieval | dense search，`top_k=5` |
| Agent | `simple` |
| Answer model | 固定 Ollama model/version |
| Temperature | 0 |
| Evaluation set | 固定 10 題 smoke set，再固定一個較大 dev set |
| Random seed | 42 |

正式跑 v1 前，必須將 dataset、embedding 同 answer model 嘅實際 revision 寫入 run config；`main` 或 floating tag 只適合早期開發，唔適合正式比較。

### 2.3 Definition of Done

- [ ] 由全新目錄成功建立 index。
- [ ] 用同一 config resume index，唔會重複或遺失 chunks。
- [ ] `ask` command 可以回答問題並輸出 document IDs。
- [ ] 固定 10 題 smoke evaluation 可以完整執行。
- [ ] 每題記錄 answer、retrieved document IDs、tool calls、latency、token usage 及 error。
- [ ] Run folder 包含完整 config、raw logs、summary 同 HTML report。
- [ ] 重跑同一設定時，資料選擇同 index manifest 一致。
- [ ] 為主要純函數及 CLI validation 加入最小測試集。

## 3. 目前進度快照（2026-08-13）

以下狀態係按現有程式碼檢視；「已實作」唔代表已用真實 dataset/model 完成驗證。

| 範圍 | 狀態 | 現有位置／備註 |
| --- | --- | --- |
| Dataset loading、normalization、chunking | 已實作，待端到端驗證 | `dataset.py` |
| Deduplication、embedding、FAISS indexing | 已實作，待端到端驗證 | `dataset.py` |
| Index manifest、resume validation | 已實作，待故障及重啟測試 | `dataset.py` |
| Structured retrieval tool | 已實作，待真實查詢驗證 | `tools.py` |
| Simple agent | 已實作，待 Ollama smoke test | `agent.py` |
| Deep agent | 已實作，但唔屬於 v1 驗收範圍 | `agent.py` |
| `index`、`ask`、`eval` CLI | 已實作，待端到端驗證 | `main.py` |
| Resumable JSONL evaluation logs | 已實作，待 interrupted-run 測試 | `eval.py` |
| Local retrieval／efficiency summary | 部分實作 | 未包括官方 answer metric 同完整 failure taxonomy |
| Automated tests | 未見 | 應先覆蓋 deterministic data/eval functions |
| Per-run visual report | 未實作 | 建議下一個小功能 |
| Official benchmark evaluation | 未實作／未驗證 | 要先確認官方 evaluator 同可比較條件 |

## 4. Step-by-step 執行程序

### Step 1：凍結實驗輸入

- 填寫 dataset revision、embedding revision、answer model version。
- 建立固定 smoke question IDs，同一輪開發唔好隨機換題。
- 保存所有參數到 run-specific `config.json`。

產物：`config.json`、question manifest、index manifest。

### Step 2：建立同驗證 index

- 先用固定 1,000 documents 建 index。
- 記錄原始 document 數、chunk 數、deduplicated 數、embedding 時間同 index size。
- 隨機抽查至少 10 個 chunks，確認內容同 metadata 可以追溯到原文件。
- 測試 resume；之後先考慮 full corpus。

產物：FAISS index、metadata store、index manifest、ingestion statistics。

### Step 3：做單題 retrieval smoke test

- 先直接測 retrieval，唔經 agent，檢查 top-5 內容同分數。
- 再用 `simple` agent 回答同一題。
- 確認答案引用嘅 document IDs 真係來自該次 tool result。

產物：一題完整 trace，包括 query、top-k chunks、分數、答案同 latency。

### Step 4：跑固定 10 題 smoke evaluation

- 執行固定題目集合。
- 檢查 error、空 retrieval、冇 tool call、無法抽取 token usage 等情況。
- 中途停止一次再 resume，確認唔會重複成功題目。

產物：`answers.jsonl`、`run_details.jsonl`、`errors.jsonl`、`summary.json`。

### Step 5：生成 v1 報告

- 驗證 raw logs schema。
- 計算 aggregate metrics 同 failure categories。
- 生成一份自包含 `report.html`，並另存 machine-readable `metrics.json`。
- 報告清楚標示呢次 run 係 smoke、dev 或 benchmark-comparable。

產物：`metrics.json`、`report.html`。

### Step 6：凍結 Baseline 0

- 將 config、code commit、dataset/index fingerprints 同環境版本寫入 manifest。
- 將 v1 設為之後比較嘅 control。
- 未通過 Definition of Done 前，唔開始優化 retrieval 或 agent。

### Step 7：逐輪實驗

每輪遵守同一個循環：

```text
提出一個假設
    ↓
只改一個主要變數
    ↓
跑同一固定 evaluation set
    ↓
生成報告及同 control 比較
    ↓
接受、拒絕或標記為不確定
    ↓
更新 version ledger
```

## 5. Data Visualization Pipeline

### 5.1 建議架構

第一階段採用 artifact-first 報告，而唔係長期運行嘅 dashboard：

```text
run config + JSONL logs + index manifest
                    ↓
             schema validation
                    ↓
        normalized per-question records
                    ↓
          derived metrics and deltas
                    ↓
       charts + tables + failure samples
                    ↓
          self-contained report.html
```

呢個做法有三個優點：run 完成後報告唔會隨外部資料改變、容易連同實驗結果封存、亦可以喺之後加 dashboard 時重用相同 normalized records。

### 5.2 MVP 報告應包含嘅圖

1. **Run health cards**：completed、failed、no-result、benchmark-comparable。
2. **Retrieval quality**：document recall 分佈，同 Recall@5 平均值。
3. **Latency distribution**：總 latency 直方圖／箱形圖，另列 retrieval latency。
4. **Tool-call distribution**：每題 tool-call 次數，找出冇檢索或過度檢索。
5. **Quality–latency scatter**：每題 retrieval quality 對 latency，找出慢但冇改善嘅 case。
6. **Failure table**：question ID、question type、錯誤分類、簡短 trace。
7. **Config panel**：dataset、models、chunking、top-k、agent、code commit。

v2 起再加入：

- **Delta cards**：相對 v1 嘅 quality、latency、tool calls 變化。
- **Per-question regression heatmap**：邊啲題目改善、退步或維持不變。
- **Pareto frontier**：比較多個 version 嘅 quality 對 latency／token usage。
- **Retrieval evidence view**：揀一題後查看 query、rank、score、chunk text、gold document 同最終引用。
- **Failure taxonomy trend**：no retrieval、wrong document、unsupported answer、agent/tool error、timeout。

### 5.3 建議 run folder

```text
runs/
  v1-simple-baseline/
    config.json
    index_manifest.json
    answers.jsonl
    run_details.jsonl
    errors.jsonl
    summary.json
    metrics.json
    report.html
  v2-hnsw-index/
    ...
```

所有圖表都必須由 run folder 入面嘅 immutable artifacts 重建，唔好直接讀 process memory。初期可以用 Plotly 生成單一 HTML；當 runs 數量多、需要跨實驗互動查詢時，先加入 DuckDB/Parquet 同 Streamlit 或其他 dashboard 層。

### 5.4 Visualization 資料欄位缺口

現有 logs 已有一部分基礎資料，但實作報告前應確認或補充：

- `run_id`、`version_id`、`parent_run_id`
- code commit SHA、環境及 dependency versions
- 每次 retrieval 嘅 ranked chunk IDs、document IDs 同 similarity scores
- index／retrieval／generation 分段 latency
- prompt、completion 同 total token counts
- question type、gold documents、retrieval metrics、answer metrics
- 統一 `error_type`，唔只保存 exception message
- 每個 artifact 嘅 schema version

## 6. Version Ledger

每次實驗開始前先填 hypothesis，完成後先填結果及決定。

| ID | 狀態 | 唯一主要改動 | Hypothesis | Primary metric | Guardrail | 結果 | 決定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | Planned | 最簡 simple-agent baseline | 建立可重現 control | Recall@5／官方 answer metric | error rate、latency | — | — |
| v2 | Backlog | FAISS exact flat index → HNSW | 加快大規模 vector search，同時保留接近 exact search 嘅 recall | Retrieval latency p50／p95 | Recall@5、build time、index size、memory | — | — |
| v3 | Backlog | `top_k: 5 → 10` | 增加 recall | Recall@k／answer quality | latency、extra docs | — | — |
| v4 | Backlog | 加入 reranker | 減少無關 evidence | answer quality／citation accuracy | latency | — | — |
| v5 | Backlog | simple → deep agent | 多步檢索改善複雜題 | answer quality by question type | tool calls、tokens、latency | — | — |

v2 已定為 HNSW index 比較；v3 之後嘅次序只係候選。完成每個 version 後，應先根據 failure report 揀最大瓶頸，唔應該預設 reranker 或 deep agent 一定係下一步。

### 6.1 v2：HNSW Vector Index

v2 會保留 v1 嘅 dataset、chunks、embeddings、`top_k=5`、agent、prompt 同 evaluation questions，只將 FAISS exact flat index 換成 HNSW approximate nearest-neighbour index。咁樣 v1 同 v2 嘅差異先可以歸因於 index algorithm。

建議第一組固定參數：

| HNSW 參數 | 初始值 | 用途 |
| --- | --- | --- |
| `M` | 32 | 控制每個 node 嘅 graph connections；越高通常 recall 越好，但 index 更大 |
| `efConstruction` | 200 | 控制建圖搜尋闊度；越高建圖越慢，但 graph quality 通常更好 |
| `efSearch` | 64 | 控制查詢搜尋闊度；越高 recall 通常越好，但 query latency 亦會增加 |

因為 v1 embeddings 已正規化並用 inner product 表示 cosine similarity，v2 必須維持同一種 scoring semantics。實作時亦要將 index type 同以上參數寫入 index manifest，防止錯誤 resume 不相容嘅 index。

v2 報告最少要比較：

- Recall@5 相對 v1 exact search 嘅差異。
- Retrieval latency p50、p95 同 p99。
- Index build time、disk size 同 peak memory。
- End-to-end answer quality、總 latency 同 error rate。

HNSW 主要解決規模同 latency 問題；1,000-document smoke corpus 只用作功能驗證，效能結論應該喺固定大型 corpus 或 full corpus 上量度。

## 7. 下一個最小工作包

- [ ] 加最小 automated tests，鎖定 selection、metrics、resume 同 config validation 行為。
- [ ] 完成 v1 嘅 1,000-document index smoke test。
- [ ] 跑固定 10 題 evaluation 並收集第一份真實 run artifacts。
- [ ] 寫 `report` command：由 run folder 生成 `metrics.json` 同 `report.html`。
- [ ] v1 驗收後實作 v2 HNSW index，並以同一 evaluation set 同 v1 exact search 比較。

## 8. 決策記錄

| 日期 | 決策 | 原因 |
| --- | --- | --- |
| 2026-08-13 | 先做 simple-agent v1 | 減少變數，先證明 end-to-end reproducibility |
| 2026-08-13 | 先做 per-run static HTML，後做 dashboard | 保持 artifacts 可封存、可重建，同時降低初期系統複雜度 |
| 2026-08-13 | v2 將 FAISS exact flat index 換成 HNSW | 比較 approximate search 對大型 corpus latency、recall、build time 同資源用量嘅影響 |
