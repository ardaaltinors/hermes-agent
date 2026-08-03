"""Tests for public Codex auth helpers used by speech-to-text UIs."""

import base64
import json
from unittest.mock import Mock, patch

from hermes_cli.auth import (
    codex_account_id_from_access_token,
    has_codex_runtime_credentials,
)


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_part({'alg': 'none'})}.{_part(claims)}.sig"


def test_codex_account_id_is_read_from_access_token_claims():
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": " account-123 "
            }
        }
    )

    assert codex_account_id_from_access_token(token) == "account-123"


def test_codex_credential_probe_prefers_singleton_without_refreshing_pool():
    pool_probe = Mock(return_value="pool-token")
    with (
        patch(
            "hermes_cli.auth._read_codex_tokens",
            return_value={"tokens": {"access_token": "singleton-token"}},
        ),
        patch("hermes_cli.auth._pool_codex_access_token", pool_probe),
    ):
        assert has_codex_runtime_credentials() is True

    pool_probe.assert_not_called()


def test_codex_credential_probe_supports_pool_only_auth():
    with (
        patch(
            "hermes_cli.auth._read_codex_tokens",
            return_value={"tokens": {}},
        ),
        patch(
            "hermes_cli.auth._pool_codex_access_token",
            return_value="pool-token",
        ),
    ):
        assert has_codex_runtime_credentials() is True


def test_codex_credential_probe_reports_missing_auth():
    with (
        patch(
            "hermes_cli.auth._read_codex_tokens",
            return_value={"tokens": {}},
        ),
        patch("hermes_cli.auth._pool_codex_access_token", return_value=""),
    ):
        assert has_codex_runtime_credentials() is False
