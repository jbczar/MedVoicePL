"""HTTP clients for downstream services.

`gateway.clients` is the ONLY package allowed to talk HTTP to
`on_premise` and `cloud` services. The trust-boundary lint enforces
this — no other gateway module may import `httpx` for outbound
purposes.
"""
