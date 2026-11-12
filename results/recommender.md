# Resolution recommender (SYNTHETIC DATA)

Index: 6305 training-period tickets (until 2025-02-06). Held out: 1287 test-period tickets (from 2025-07-21). Query: `{module} | {short_description}`. A hit = a retrieved past ticket has the same category (8 categories).

| retriever | hit@1 | hit@5 | precision@5 | MRR |
|---|---|---|---|---|
| tfidf | 0.908 | 0.943 | 0.888 | 0.920 |
| minilm | not available: could not load sentence-transformers/all-MiniLM-L6-v2: OSError | | | |
| minilm_onnx | not available: no ONNX model cached and no archive at /sessions/rcw-011nvq6j4cmepd1mec2dp3r6/mnt/capstone/sap-incident-forecasting/models/onnx.tar.gz | | | |
| random_any_ticket (expected) | 0.158 | 0.546 | 0.158 | - |
| random_same_module (expected) | 0.183 | 0.588 | 0.183 | - |
