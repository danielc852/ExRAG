# Known Problem: OpenRouter Free-Request Limit

## Summary

The current OpenRouter account is limited to 50 free-model requests per day.
The limit is shared by all `:free` models on the account, so embedding requests
and LLM generation requests consume the same daily allowance.

OpenRouter reported the limit directly in an HTTP 429 response:

```text
Rate limit exceeded: free-models-per-day.
Limit: 50
Remaining: 0
```

This allowance and its reset time are controlled by OpenRouter and may change.
Always treat the response headers as the authoritative source.

## Impact on the Sample Experiment

The sample index contains 4,548 chunks. With an embedding batch size of 100, a
complete rebuild needs approximately:

```text
ceil(4,548 / 100) = 46 embedding requests
```

Additional requests were used by smoke tests and earlier rebuild attempts. This
exhausted the 50-request daily allowance before the 10-question LLM experiment
started. Consequently, all 10 questions received HTTP 429 responses and the run
produced no valid benchmark results.

The failed diagnostic run is stored in:

```text
runs/sample/openrouter-nemotron35-liquid-20260819-simple/
```

Its summary is `0 completed / 10 failed`. It must not be used as a benchmark
result.

## Why Retrying Does Not Solve It

Short-term rate limits can be handled with `Retry-After` and bounded backoff.
The daily free-model limit is different: retries continue to fail until the
allowance resets or the account limit changes. A daily-limit response should
therefore stop the operation instead of repeatedly retrying.

## Safe Operating Procedure

1. Reuse the completed index whenever its embedding configuration matches the
   experiment. Do not rebuild it before every run.
2. On a fresh daily allowance, run either a full embedding rebuild or the sample
   LLM experiment, not both.
3. Use a large supported embedding batch size to reduce request count. The
   current sample index was built with a batch size of 100.
4. Before a rebuild, estimate requests as `ceil(chunk_count / batch_size)` and
   leave capacity for smoke tests, query embeddings, and LLM calls.
5. When HTTP 429 reports `free-models-per-day`, read `X-RateLimit-Reset` and wait
   for that time. Do not repeatedly restart the pipeline.
6. If same-day rebuilds and experiments are required, add OpenRouter credits or
   use models and billing that are not constrained by the 50-request free tier.

## Running the Experiment Without Rebuilding

After the daily allowance resets, run only the experiment:

```bash
uv run --env-file .env python main.py run_exper sample \
  --output-dir runs/sample/openrouter-nemotron35-liquid-YYYYMMDD-simple \
  --no-resume
```

The `.env` configuration selects:

```text
LLM provider: OpenRouter
LLM model: nvidia/nemotron-3.5-lightning:free
Embedding engine: OpenRouter
Embedding model: liquid/lfm-2.5-embedding-350m:free
```

## Completed Index Configuration

```text
Dataset: sample (1,000 documents)
Chunks/vectors: 4,548
Embedding dimension: 1,024
Tokenizer: LiquidAI/LFM2.5-Embedding-350M
Chunk size: 400
Chunk overlap: 48
Embedding batch size: 100
```

The completed index should be retained and reused unless the dataset, tokenizer,
chunking configuration, or embedding model changes.
