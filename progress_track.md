# EnterpriseRAG Progress Track

用 `Day 0`、`Day 1`、`Day 2` 順序記錄實際工作日；每個 Day 同時寫低日期。

每次只記五項：

1. Goal
2. Progress
3. Decisions
4. Debug
5. Next

---

## Day 0 — 2026-08-16

### Goal

建立可運行嘅 EnterpriseRAG baseline，並簡化 agent、tools、data 同 evaluation 架構。

### Progress

- 建立 `agent/` package：agent factory、LLM、runtime、state、helpers。
- 建立 `tools/get_docs/`：schema、tool implementation、agent description。
- 建立 `eval/`：local evaluation、LangSmith sync／run／compare。
- 將 `data/` 分成：
  - `preprocessing/`：download、review、clean text。
  - `processing/`：chunk、embed、store、index。
- 將 chunking 從 cleaning 拆出，並將 `data_cleaning.py` 改名為 `clean_data.py`。
- 準備 10 題 EnterpriseRAG-Bench JSONL sample。
- 今日最後測試結果：`42 passed, 2 skipped`。

### Decisions

- Local evaluation 同 LangSmith evaluation 兩條流程並存。
- Agent 沿用 LangChain `Runnable`，暫時唔建立自訂 ABC。
- 每個 tool 使用獨立 folder，分開 schema、code 同 description。
- Pipeline 使用簡單名稱：`pre_data()`、`pre_store()`、`run_process()`、`get_status()`。
- Chunking 屬於 vector processing，唔屬於 text cleaning。
- 本地 benchmark 使用 Parquet + FAISS + SQLite，暫時唔使用 PostgreSQL。
- 保留原文同 metadata；embedding 唔可以取代原文。
- 先完成可重現 v1 baseline，再考慮 HNSW 或進一步簡化 manifests。

### Debug

#### Open

- Local evaluation resume 未完整檢查 `top_k`、index fingerprint 同 question IDs。
- Truncated JSONL 同 `answers`／`run_details` 分開寫入可能令 resume 資料不一致。
- LangSmith run 未重新驗證 cloud examples；experiment comparison 亦未拒絕缺失 metadata。
- Final index validation 未完整驗證 FAISS／SQLite checksum。
- 第一份 LangSmith JSONL mapping 錯誤；修正版已建立，但仍要刪除舊 10 題再重新匯入。
- Chrome extension 未有 file URL 權限，因此 sample upload 未完成。

#### Resolved

- 已刪除只剩 cache 嘅舊 `tools/retrieve_documents/` folder。
- LangSmith JSONL 已改用 `inputs`、`outputs`、`metadata` 結構。

### Next

- [ ] 修正 evaluation resume、JSONL recovery 同 cloud validation 問題。
- [ ] 加入 final FAISS／SQLite integrity check。
- [ ] 重新匯入 10 題 LangSmith sample，確認 inputs 同 reference outputs。
- [ ] 建立固定 1,000-document index。
- [ ] 跑固定 10 題 end-to-end evaluation，保存第一份完整 run artifacts。

### Commits

`4fda2f2` → `c5c3aed` → `55fd9cf` → `f1e814e` → `85d3179` → `c325a46` → `5ebe3c0` → `9aa5c36` → `9bc8d5d` → `58f2436` → `516cdc4`

---

## Day 1 — YYYY-MM-DD

### Goal

-

### Progress

-

### Decisions

-

### Debug

- Open:
- Resolved:

### Next

- [ ]
