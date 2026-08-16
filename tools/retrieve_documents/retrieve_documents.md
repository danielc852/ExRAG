Search enterprise documents for evidence relevant to a question.

Use this tool before answering questions about enterprise information. Base the answer on the returned content and preserve the document and chunk IDs as citation evidence.

Only set `source_types` or `document_ids` when the user explicitly supplies those constraints. Never invent filters. Increase `top_k` only when the initial evidence is incomplete or the question requires multiple sources.
