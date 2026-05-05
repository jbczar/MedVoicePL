"""Auth middleware moved into `gateway.api.auth.require_session` dependency.

This stub is kept for backwards compatibility with the planned import
path. Use the `require_session` dependency in routers instead — it
gives FastAPI dependency injection a chance to override in tests.
"""
