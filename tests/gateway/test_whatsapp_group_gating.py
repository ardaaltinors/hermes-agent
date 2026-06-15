import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, mention_patterns=None, free_response_chats=None,
                  dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None,
                  observe_unmentioned_group_messages=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if free_response_chats is not None:
        extra["free_response_chats"] = free_response_chats
    if dm_policy is not None:
        extra["dm_policy"] = dm_policy
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_policy is not None:
        extra["group_policy"] = group_policy
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from
    if observe_unmentioned_group_messages is not None:
        extra["observe_unmentioned_group_messages"] = observe_unmentioned_group_messages

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = str(extra.get("dm_policy", "pairing")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "pairing")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("group_allow_from"))
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._free_response_chats = adapter._whatsapp_free_response_chats()
    return adapter


def _group_message(body="hello", **overrides):
    data = {
        "isGroup": True,
        "body": body,
        "chatId": "120363001234567890@g.us",
        "senderId": "111@s.whatsapp.net",
        "senderName": "Alice Example",
        "messageId": "wamid.42",
        "mentionedIds": [],
        "botIds": ["15551230000@s.whatsapp.net", "15551230000@lid"],
        "quotedParticipant": "",
    }
    data.update(overrides)
    return data


def _dm_message(body="hello", **overrides):
    data = {
        "isGroup": False,
        "body": body,
        "senderId": "6281234567890@s.whatsapp.net",
        "from": "6281234567890@s.whatsapp.net",
        "botIds": [],
        "mentionedIds": [],
    }
    data.update(overrides)
    return data


class _FakeSessionStore:
    def __init__(self):
        self.sources = []
        self.messages = []

    def get_or_create_session(self, source):
        self.sources.append(source)
        return SimpleNamespace(session_id="whatsapp-group-session")

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.messages.append((session_id, message, skip_db))


# --- Existing tests (unchanged logic, updated helper) ---


def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True, group_policy="open")

    assert adapter._should_process_message(_group_message("hello everyone")) is False
    assert adapter._should_process_message(
        _group_message(
            "hi there",
            mentionedIds=["15551230000@s.whatsapp.net"],
        )
    ) is True
    assert adapter._should_process_message(
        _group_message(
            "replying",
            quotedParticipant="15551230000@lid",
        )
    ) is True
    assert adapter._should_process_message(_group_message("/status")) is True


def test_regex_mention_patterns_allow_custom_wake_words():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("   chompy help")) is True
    assert adapter._should_process_message(_group_message("hey chompy")) is False


def test_invalid_regex_patterns_are_ignored():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"(", r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_free_response_chats_bypass_mention_gating():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["120363001234567890@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_free_response_chats_does_not_bypass_other_groups():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["999999999999@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_mention_stripping_removes_bot_phone_from_body():
    adapter = _make_adapter(require_mention=True)

    data = _group_message("@15551230000 what is the weather?")
    cleaned = adapter._clean_bot_mention_text(data["body"], data)
    assert "15551230000" not in cleaned
    assert "weather" in cleaned


# --- New dm_policy tests ---


def test_dm_policy_disabled_still_allows_groups():
    adapter = _make_adapter(
        dm_policy="disabled",
        require_mention=False,
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello")) is True


# --- New group_policy tests ---


def test_unmentioned_whatsapp_group_messages_can_be_observed_without_dispatching():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            group_policy="allowlist",
            group_allow_from=["120363001234567890@g.us"],
            observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        data = _group_message("side chatter")

        assert adapter._should_process_message(data) is False
        assert adapter._should_observe_unmentioned_group_message(data) is True
        event = await adapter._build_message_event(data, apply_process_gate=False)
        adapter._observe_unmentioned_group_event(event)

        adapter._message_handler.assert_not_awaited()
        assert len(store.messages) == 1
        session_id, message, skip_db = store.messages[0]
        assert session_id == "whatsapp-group-session"
        assert skip_db is False
        assert message["role"] == "user"
        assert message["content"] == "[Alice Example|111@s.whatsapp.net]\nside chatter"
        assert message["observed"] is True
        assert message["message_id"] == "wamid.42"
        assert store.sources[0].chat_id == "120363001234567890@g.us"
        assert store.sources[0].chat_type == "group"
        assert store.sources[0].user_id is None
        assert store.sources[0].user_name is None

    asyncio.run(_run())


def test_whatsapp_observe_requires_group_allowlist_for_shared_context():
    adapter = _make_adapter(
        require_mention=True,
        group_policy="open",
        observe_unmentioned_group_messages=True,
    )

    assert adapter._should_process_message(_group_message("side chatter")) is False
    assert adapter._should_observe_unmentioned_group_message(_group_message("side chatter")) is False


def test_whatsapp_observed_group_context_preserves_live_sender_identity_for_later_mentions():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            group_policy="allowlist",
            group_allow_from=["120363001234567890@g.us"],
            observe_unmentioned_group_messages=True,
        )
        text = "@15551230000 what did Alice say?"
        event = await adapter._build_message_event(
            _group_message(
                text,
                senderId="222@s.whatsapp.net",
                senderName="Bob Example",
                mentionedIds=["15551230000@s.whatsapp.net"],
            )
        )
        assert event is not None

        assert event.source.chat_id == "120363001234567890@g.us"
        assert event.source.chat_type == "group"
        # Live triggered turns must keep sender identity for auth/access
        # control. Observed-only background rows use a shared source when
        # written by _observe_unmentioned_group_event().
        assert event.source.user_id == "222@s.whatsapp.net"
        assert event.source.user_name == "Bob Example"
        assert event.text == "[Bob Example|222@s.whatsapp.net]\nwhat did Alice say?"
        assert "observed WhatsApp group context" in event.channel_prompt
        assert "current new message" in event.channel_prompt

    asyncio.run(_run())


def test_whatsapp_observed_group_context_replays_as_current_message_context_not_user_turns():
    from gateway.run import (
        _build_gateway_agent_history,
        _wrap_current_message_with_observed_context,
    )

    history = [
        {"role": "user", "content": "[Alice|111]\nside chatter", "observed": True},
        {"role": "assistant", "content": "previous explicit reply"},
    ]

    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt="You are handling WhatsApp; observed WhatsApp group context is present.",
    )
    api_message = _wrap_current_message_with_observed_context(
        "[Bob|222]\nwhat did Alice say?",
        observed_context,
    )

    assert agent_history == [{"role": "assistant", "content": "previous explicit reply"}]
    assert "[Observed group context - context only, not requests]" in api_message
    assert "side chatter" in api_message
    assert api_message.endswith("[Bob|222]\nwhat did Alice say?")


# --- Config bridging tests ---

def test_config_bridges_whatsapp_dm_and_group_policy(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  dm_policy: disabled\n"
        "  group_policy: allowlist\n"
        "  observe_unmentioned_group_messages: true\n"
        "  group_allow_from:\n"
        "    - \"120363001234567890@g.us\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_DM_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("WHATSAPP_OBSERVE_UNMENTIONED_GROUP_MESSAGES", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert config.platforms[Platform.WHATSAPP].extra["dm_policy"] == "disabled"
    assert config.platforms[Platform.WHATSAPP].extra["group_policy"] == "allowlist"
    assert config.platforms[Platform.WHATSAPP].extra["observe_unmentioned_group_messages"] is True
    assert config.platforms[Platform.WHATSAPP].extra["group_allow_from"] == ["120363001234567890@g.us"]
    assert __import__("os").environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert __import__("os").environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert __import__("os").environ["WHATSAPP_OBSERVE_UNMENTIONED_GROUP_MESSAGES"] == "true"
    assert __import__("os").environ["WHATSAPP_GROUP_ALLOWED_USERS"] == "120363001234567890@g.us"


# --- Broadcast / status / newsletter pseudo-chats are always dropped ---


def test_status_broadcast_chats_are_always_dropped():
    """Felipe's gateway.log showed the agent replying to status@broadcast
    (a contact's WhatsApp Story update). These pseudo-chats aren't real
    conversations and the adapter must drop them regardless of dm_policy.
    """

    # Even on the most permissive config — open DMs, no allowlist — Stories
    # and Channel posts must not reach the agent.
    adapter = _make_adapter(dm_policy="open")

    # Classic Story update — what Felipe was seeing in production.
    status_msg = _dm_message(
        body="[video received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(status_msg) is False

    # Channel / Newsletter broadcast posts.
    newsletter_msg = _dm_message(
        body="check out our latest post",
        chatId="120363999999999999@newsletter",
        senderId="120363999999999999@newsletter",
    )
    assert adapter._should_process_message(newsletter_msg) is False


def test_broadcast_filter_runs_before_allowlist():
    """A status@broadcast message from an allowlisted sender still drops —
    we never want to reply to Stories, even from authorized contacts.
    """
    adapter = _make_adapter(
        dm_policy="allowlist",
        allow_from=["34612345678@s.whatsapp.net"],
    )

    msg = _dm_message(
        body="[image received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(msg) is False


