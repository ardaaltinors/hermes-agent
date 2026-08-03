"""Behavior tests for ChatGPT/Codex OAuth speech-to-text."""

from pathlib import Path
from unittest.mock import Mock, patch


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            error = requests.HTTPError(f"{self.status_code} response")
            error.response = self
            raise error


def test_explicit_openai_codex_provider_uses_existing_oauth_login():
    from tools.transcription_tools import _get_provider

    with patch(
        "tools.transcription_tools._resolve_codex_stt_credentials",
        return_value={"api_key": "oauth-token"},
    ):
        assert _get_provider({"provider": "openai-codex"}) == "openai-codex"


def test_codex_credentials_include_optional_chatgpt_account_id():
    from tools.transcription_tools import _resolve_codex_stt_credentials

    with (
        patch(
            "hermes_cli.auth.resolve_codex_runtime_credentials",
            return_value={"api_key": "oauth-token", "source": "hermes-auth-store"},
        ),
        patch(
            "hermes_cli.auth._read_codex_tokens",
            return_value={"tokens": {"account_id": "account-123"}},
        ),
    ):
        credentials = _resolve_codex_stt_credentials()

    assert credentials["account_id"] == "account-123"


def test_openai_codex_transcription_uses_subscription_endpoint(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.webm"
    audio.write_bytes(b"audio")
    response = _Response(payload={"text": "Merhaba Codex"})

    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials",
            return_value={"api_key": "oauth-token", "account_id": "account-123"},
        ),
        patch("requests.post", return_value=response) as post,
    ):
        result = _transcribe_openai_codex(str(audio), language="tr")

    assert result == {
        "success": True,
        "transcript": "Merhaba Codex",
        "provider": "openai-codex",
    }
    kwargs = post.call_args.kwargs
    assert post.call_args.args[0] == "https://chatgpt.com/backend-api/transcribe"
    assert kwargs["headers"]["Authorization"] == "Bearer oauth-token"
    assert kwargs["headers"]["ChatGPT-Account-Id"] == "account-123"
    assert kwargs["headers"]["User-Agent"].startswith("codex-cli")
    assert kwargs["data"] == {"language": "tr"}
    filename, handle, mime = kwargs["files"]["file"]
    assert filename == "voice.webm"
    assert mime == "audio/webm"
    assert handle.closed


def test_openai_codex_transcription_refreshes_once_after_unauthorized(tmp_path):
    from tools.transcription_tools import _transcribe_openai_codex

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    resolver = Mock(
        side_effect=[{"api_key": "old-token"}, {"api_key": "new-token"}]
    )

    with (
        patch(
            "tools.transcription_tools._resolve_codex_stt_credentials", resolver
        ),
        patch(
            "requests.post",
            side_effect=[
                _Response(status_code=401, text="expired"),
                _Response(payload={"text": "yenilendi"}),
            ],
        ) as post,
    ):
        assert _transcribe_openai_codex(str(audio)) == {
            "success": True,
            "transcript": "yenilendi",
            "provider": "openai-codex",
        }

    assert resolver.call_args_list[0].kwargs == {}
    assert resolver.call_args_list[1].kwargs == {"force_refresh": True}
    assert post.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer new-token"


def test_dispatch_routes_openai_codex_provider(tmp_path):
    from tools.transcription_tools import _transcribe_prepared_audio

    audio = tmp_path / "voice.webm"
    audio.write_bytes(b"audio")
    config = {
        "enabled": True,
        "provider": "openai-codex",
        "language": "tr",
        "openai_codex": {"timeout": 45},
    }

    with (
        patch("tools.transcription_tools._load_stt_config", return_value=config),
        patch("tools.transcription_tools._get_provider", return_value="openai-codex"),
        patch(
            "tools.transcription_tools._transcribe_openai_codex",
            return_value={
                "success": True,
                "transcript": "uçtan uca",
                "provider": "openai-codex",
            },
        ) as transcribe,
    ):
        assert _transcribe_prepared_audio(str(audio)) == {
            "success": True,
            "transcript": "uçtan uca",
            "provider": "openai-codex",
        }

    transcribe.assert_called_once_with(str(audio), language="tr", timeout=45)


def test_dispatch_preserves_explicit_empty_codex_language_for_auto_detect(tmp_path):
    from tools.transcription_tools import _transcribe_prepared_audio

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    config = {
        "enabled": True,
        "provider": "openai-codex",
        "language": "en",
        "openai_codex": {"language": "", "timeout": 120},
    }

    with (
        patch("tools.transcription_tools._load_stt_config", return_value=config),
        patch("tools.transcription_tools._get_provider", return_value="openai-codex"),
        patch(
            "tools.transcription_tools._transcribe_openai_codex",
            return_value={
                "success": True,
                "transcript": "Türkçe",
                "provider": "openai-codex",
            },
        ) as transcribe,
    ):
        result = _transcribe_prepared_audio(str(audio))

    assert result["success"] is True
    transcribe.assert_called_once_with(str(audio), language="", timeout=120)
