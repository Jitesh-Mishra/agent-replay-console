---
queue: Technical Support
title: API and integration errors
synthetic: true   # self-authored for this project, not from any company
---

# API and integration errors

For integration errors collect the request id, timestamp, endpoint, and response code. Rate limits return 429; advise exponential backoff. Authentication errors usually mean an expired token or a revoked key.
