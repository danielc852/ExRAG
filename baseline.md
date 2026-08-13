# EnterpriseRAG-Bench Baseline

## 1. 目標

建立一個簡單、可重現、方便比較嘅 Retrieval-Augmented Generation（RAG）baseline，用 EnterpriseRAG-Bench 做測試。系統提供兩種 agent 選項，但兩者共用同一套文件處理及 retrieval pipeline，避免比較時引入額外變數。

## 2. 系統組件

```text
EnterpriseRAG-Bench corpus
        ↓
文件載入與標準化
        ↓
切分 chunks
        ↓
過濾、建立 embeddings、索引
        ↓
Retrieval tool
        ↓
Simple LangChain agent / Deep agent
        ↓
答案、引用來源及 evaluation logs
```

### 2.1 Agent 選項 A：LangChain simple tool-calling agent

最小化 agent baseline，只提供必要工具，等模型按問題決定何時檢索文件。

- Agent framework：LangChain
- Agent 類型：tool-calling agent
- 必要工具：`retrieve_documents`
- 建議限制：每題最多 3 次 tool calls
- 輸出：最終答案、引用嘅 document/chunk IDs、tool-call trace
- 無足夠證據時：明確回覆資料不足，唔應該自行補作事實

### 2.2 Agent 選項 B：Deep agent

用 deep agent 作第二個 baseline，容許 agent 規劃多步檢索、改寫 query，同整合多個來源。

- 使用同一個 `retrieve_documents` 工具及同一個 index
- 容許多步推理及多輪 retrieval
- 建議限制：每題最多 8 個 agent steps
- 記錄 plan、tool calls、retrieved chunk IDs、token usage 同總耗時
- 除 agent orchestration 外，其他模型與 retrieval 設定應同選項 A 保持一致

## 3. 文件處理 Pipeline

### 3.1 Dataset

- Dataset：EnterpriseRAG-Bench
- 固定並記錄 dataset version、下載日期及來源
- 保留官方 train/development/test split；如官方冇提供 split，就建立一份固定、可重現嘅 manifest
- 將 corpus、questions、reference answers/qrels 分開儲存
- 每份文件保留穩定嘅 `document_id`，每個問題保留 `query_id`
- 測試集答案或 relevance labels 唔可以用作 chunking、prompt 或 retrieval 調參

每份文件最少保留以下 metadata：

| 欄位 | 用途 |
| --- | --- |
| `document_id` | 原始文件唯一識別碼 |
| `source` | 文件來源或檔案路徑 |
| `title` | 文件標題 |
| `section` | chunk 所屬章節 |
| `page` | 頁碼，如適用 |
| `dataset_split` | train、development 或 test |
| `chunk_id` | chunk 唯一識別碼 |

### 3.2 Chunking method

首個 baseline 使用固定大小、帶 overlap 嘅 recursive text splitting：

- 方法：`RecursiveCharacterTextSplitter`
- Chunk size：512 tokens
- Chunk overlap：64 tokens
- 優先分隔：章節 → 段落 → 句子 → 空白
- Token counting：盡量使用 embedding model 對應嘅 tokenizer
- `chunk_id` 格式：`{document_id}::{chunk_number}`
- 每個 chunk 必須保留原始文件 metadata

固定 chunking 參數後先比較 agent；如要研究 chunking，另開實驗比較 256、512、1,024 tokens，唔好同時改其他變數。

### 3.3 Filtering / indexing

#### Pre-index filtering

- 移除空白、只包含導覽文字或低資訊量嘅 chunks
- 統一 Unicode、換行及多餘空白
- 用內容 hash 移除完全重複 chunks
- 唔應該移除數字、表格內容或企業專用詞
- 記錄每份文件原始及保留嘅 chunk 數量，方便審計

#### Indexing

- Baseline vector store：FAISS（本地、簡單、可重現）
- Similarity：cosine similarity
- Retrieval：dense vector search
- Default `top_k`：5
- Index 必須連同 dataset version、chunking config、embedding model name/version 一齊標記
- Dataset 或任何 indexing 設定有改動時，重新建立 index

Metadata filters 應屬可選參數，例如 `document_id`、`source` 或 `section`；baseline 唔應該用 test labels 做 filtering。

### 3.4 Embedding model

建議預設：`BAAI/bge-base-en-v1.5`

- 固定 model revision，避免同名模型更新後令結果漂移
- 文件同 query 使用同一個 embedding model
- 按模型要求正規化 embeddings
- 記錄 embedding dimension、batch size、device 同執行時間
- 如果 corpus 包含大量非英文內容，可以另設 multilingual experiment，例如 `BAAI/bge-m3`，但唔應該同英文 baseline 結果直接混合

## 4. Retrieval Tool

兩種 agent 共用同一個工具介面：

```python
retrieve_documents(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[RetrievedChunk]
```

每個 `RetrievedChunk` 應包括：

```text
chunk_id
document_id
content
title
section/page（如有）
similarity_score
source
```

工具行為：

1. 驗證 query 唔係空字串。
2. 將 query 轉成 embedding。
3. 套用獲允許嘅 metadata filters。
4. 從 vector index 取回 top-k chunks。
5. 以結構化格式回傳內容、分數及 citation metadata。
6. 將 query、filters、結果 IDs、分數及 latency 寫入 evaluation log。

## 5. 建議 Baseline Configuration

```yaml
dataset:
  name: EnterpriseRAG-Bench
  version: "PINNED_VERSION"
  split: test

chunking:
  method: recursive
  chunk_size_tokens: 512
  chunk_overlap_tokens: 64

filtering:
  normalize_unicode: true
  collapse_whitespace: true
  remove_empty_chunks: true
  deduplicate_exact_chunks: true

embedding:
  model: BAAI/bge-base-en-v1.5
  revision: "PINNED_REVISION"
  normalize: true

index:
  vector_store: faiss
  similarity: cosine

retrieval:
  top_k: 5

agent:
  type: langchain_tool_calling # 或 deep_agent
  max_tool_calls: 3
  temperature: 0
```

`PINNED_VERSION` 同 `PINNED_REVISION` 必須喺第一次正式實驗前填妥。

## 6. 公平比較規則

比較兩種 agent 時，以下項目必須固定：

- EnterpriseRAG-Bench version 及 evaluation split
- Chunking、filtering、embedding 及 index
- Retrieval `top_k`
- Answer model 及 model version
- System prompt 核心規則
- Temperature、context window 同輸出格式
- 執行環境及 random seed（如適用）

唯一主要變數應該係 agent orchestration：simple tool calling 對 deep agent。

## 7. 最低限度 Evaluation 指標

- Retrieval：Recall@5、MRR 或 nDCG@5（視乎 dataset labels）
- Answer quality：使用 EnterpriseRAG-Bench 官方 metric
- Grounding：答案有冇獲 retrieved evidence 支持
- Citation accuracy：引用嘅 chunk/document 係咪正確
- Efficiency：平均 latency、tool-call 次數、input/output tokens
- Reliability：錯誤率、無結果率、超出 step limit 嘅比例

所有 run 應保存完整 config、prompt version、model version、retrieval trace 同結果檔案，確保實驗可以重現。

## 8. 實作及運行

本 repository 提供兩個共用同一 retrieval index 嘅 agent：

- `simple`：LangChain tool-calling agent
- `deep`：Deep Agents SDK，使用 ephemeral state，唔存取 host filesystem

預設使用 Ollama `qwen3:8b`，而 embedding model 為 `BAAI/bge-base-en-v1.5`。第一次運行前先安裝依賴及模型：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ollama pull qwen3:8b
```

建立開發用 1,000-document sample index：

```bash
python main.py index
```

正式 benchmark 必須建立完整 corpus index：

```bash
python main.py index --full --rebuild
```

單題問答及批量 evaluation：

```bash
python main.py ask "What is the relevant policy?" --agent simple
python main.py ask "What is the relevant policy?" --agent deep
python main.py eval --agent simple
python main.py eval --agent deep --all-questions
```

`eval` 會輸出官方相容嘅 `answers.jsonl`，以及本地 retrieval、latency、tool-call 同錯誤統計。Sample index 嘅結果只供 smoke test，會標示為不可同正式 benchmark 比較。
