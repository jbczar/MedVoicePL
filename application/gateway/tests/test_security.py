"""Tests for bcrypt password hashing/verification."""

from __future__ import annotations

from gateway.security import hash_password, verify_password


def test_hash_and_verify_round_trip() -> None:
    h = hash_password("demo_password")
    assert verify_password("demo_password", h) is True


def test_verify_rejects_wrong_password() -> None:
    h = hash_password("demo_password")
    assert verify_password("wrong", h) is False


def test_verify_returns_false_for_empty_hash() -> None:
    assert verify_password("anything", "") is False


def test_verify_returns_false_for_corrupt_hash() -> None:
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_hash_produces_distinct_salts() -> None:
    a = hash_password("demo_password")
    b = hash_password("demo_password")
    assert a != b
    # Both still verify.
    assert verify_password("demo_password", a)
    assert verify_password("demo_password", b)
