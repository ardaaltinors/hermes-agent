"""Tests for constrained file uploads through the browser tool."""

import json
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools import browser_tool


@pytest.fixture(autouse=True)
def _agent_browser_mode(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_tool, "_get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: False)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")


def _allow_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(browser_tool, "_get_upload_allowed_roots", lambda: [root.resolve()])


class TestBrowserUploadSchemaAndWiring:
    def test_schema_accepts_selector_and_files(self):
        schema = next(
            item for item in browser_tool.BROWSER_TOOL_SCHEMAS
            if item["name"] == "browser_upload"
        )

        properties = schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["selector", "files"]
        assert properties["selector"]["type"] == "string"
        assert properties["files"]["type"] == "array"
        assert properties["files"]["items"]["type"] == "string"
        assert properties["files"]["minItems"] == 1

    def test_registered_in_every_browser_surface(self):
        from acp_adapter.tools import TOOL_KIND_MAP, _POLISHED_TOOLS
        from agent.tool_guardrails import MUTATING_TOOL_NAMES
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        from model_tools import _LEGACY_TOOLSET_MAP
        from tools.registry import registry
        from toolsets import TOOLSETS, _HERMES_CORE_TOOLS

        assert "browser_upload" in registry._tools
        assert "browser_upload" in TOOLSETS["browser"]["tools"]
        assert "browser_upload" in _HERMES_CORE_TOOLS
        assert "browser_upload" in _LEGACY_TOOLSET_MAP["browser_tools"]
        assert "browser_upload" in MUTATING_TOOL_NAMES
        assert TOOL_KIND_MAP["browser_upload"] == "execute"
        assert "browser_upload" in _POLISHED_TOOLS
        assert "browser_upload" in EXPOSED_TOOLS


class TestBrowserUploadAllowedRoots:
    def test_reads_and_resolves_configured_roots(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"upload_allowed_roots": [str(root)]}},
        )

        assert browser_tool._get_upload_allowed_roots() == [root.resolve()]

    def test_no_configured_roots_disables_upload(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr(browser_tool, "check_browser_requirements", lambda: True)

        assert browser_tool._get_upload_allowed_roots() == []
        assert browser_tool.check_browser_upload_requirements() is False

    def test_missing_agent_browser_disables_upload_with_cdp_override(
        self, monkeypatch, tmp_path,
    ):
        root = tmp_path / "staging"
        root.mkdir()
        _allow_root(monkeypatch, root)
        monkeypatch.setattr(browser_tool, "check_browser_requirements", lambda: True)

        def missing_cli(*_args, **_kwargs):
            raise FileNotFoundError("agent-browser is not installed")

        monkeypatch.setattr(browser_tool, "_find_agent_browser", missing_cli)

        assert browser_tool.check_browser_upload_requirements() is False

    def test_inside_root_uploads_all_files_with_resolved_paths(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        first = root / "cv final.pdf"
        second = root / "portfolio.txt"
        first.write_bytes(b"pdf")
        second.write_text("portfolio", encoding="utf-8")
        _allow_root(monkeypatch, root)
        calls = []

        def fake_run(task_id, command, args):
            calls.append((task_id, command, args))
            return {"success": True, "data": {}}

        monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

        out = json.loads(browser_tool.browser_upload(
            "input[type=file]",
            [str(first), str(second)],
            task_id="apply-1",
        ))

        assert out == {
            "success": True,
            "element": "input[type=file]",
            "uploaded": ["cv final.pdf", "portfolio.txt"],
        }
        assert calls == [(
            "apply-1",
            "upload",
            ["input[type=file]", str(first.resolve()), str(second.resolve())],
        )]

    @pytest.mark.parametrize(
        "kind", ["outside", "traversal", "missing", "directory", "invalid"],
    )
    def test_rejects_invalid_or_out_of_root_paths(self, monkeypatch, tmp_path, kind):
        root = tmp_path / "staging"
        root.mkdir()
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(b"pdf")
        _allow_root(monkeypatch, root)

        if kind == "outside":
            candidate = outside
        elif kind == "traversal":
            candidate = root / ".." / "outside.pdf"
        elif kind == "missing":
            candidate = root / "missing.pdf"
        elif kind == "invalid":
            candidate = Path("\0")
        else:
            candidate = root

        def fail_run(*_args, **_kwargs):
            raise AssertionError("agent-browser must not run for a rejected path")

        monkeypatch.setattr(browser_tool, "_run_browser_command", fail_run)

        out = json.loads(browser_tool.browser_upload(
            "#cv",
            [str(candidate)],
            task_id="apply-1",
        ))

        assert out["success"] is False
        assert "upload" in out["error"].lower() or "file" in out["error"].lower()
        assert str(outside.resolve()) not in json.dumps(out)

    def test_rejects_symlink_escape(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(b"pdf")
        link = root / "cv.pdf"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are unavailable on this platform")
        _allow_root(monkeypatch, root)

        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *_args, **_kwargs: pytest.fail("symlink escape reached agent-browser"),
        )

        out = json.loads(browser_tool.browser_upload("#cv", [str(link)], task_id="apply-1"))

        assert out["success"] is False
        assert "allowed" in out["error"].lower()

    def test_validates_every_file_before_uploading(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        allowed = root / "cv.pdf"
        allowed.write_bytes(b"pdf")
        outside = tmp_path / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        _allow_root(monkeypatch, root)

        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *_args, **_kwargs: pytest.fail("partial validation reached agent-browser"),
        )

        out = json.loads(browser_tool.browser_upload(
            "#cv",
            [str(allowed), str(outside)],
            task_id="apply-1",
        ))

        assert out["success"] is False
        assert "allowed" in out["error"].lower()


class TestBrowserUploadSafetyGuards:
    def test_rejects_invalid_selector_before_subprocess(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        cv = root / "cv.pdf"
        cv.write_bytes(b"pdf")
        _allow_root(monkeypatch, root)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *_args, **_kwargs: pytest.fail("invalid selector reached agent-browser"),
        )

        out = json.loads(browser_tool.browser_upload("#cv\0", [str(cv)], task_id="apply-1"))

        assert out["success"] is False
        assert "selector" in out["error"].lower()

    def test_private_page_blocks_upload_before_subprocess(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        cv = root / "cv.pdf"
        cv.write_bytes(b"pdf")
        _allow_root(monkeypatch, root)
        private_url = "http://169.254.169.254/latest/meta-data/"
        monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda task_id: True)
        monkeypatch.setattr(browser_tool, "_current_page_private_url", lambda task_id: private_url)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *_args, **_kwargs: pytest.fail("private page upload reached agent-browser"),
        )

        out = json.loads(browser_tool.browser_upload("#cv", [str(cv)], task_id="apply-1"))

        assert out["success"] is False
        assert private_url in out["error"]
        assert "upload" in out["error"].lower()

    def test_camofox_fails_closed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)

        out = json.loads(browser_tool.browser_upload("#cv", [str(tmp_path / "cv.pdf")]))

        assert out["success"] is False
        assert "camofox" in out["error"].lower()
        assert "not supported" in out["error"].lower()

    def test_lightpanda_fails_closed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "lightpanda")
        monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: True)

        out = json.loads(browser_tool.browser_upload("#cv", [str(tmp_path / "cv.pdf")]))

        assert out["success"] is False
        assert "lightpanda" in out["error"].lower()
        assert "not supported" in out["error"].lower()

    def test_cloud_backend_fails_closed(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        cv = root / "cv.pdf"
        cv.write_bytes(b"pdf")
        _allow_root(monkeypatch, root)
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: object())
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *_args, **_kwargs: pytest.fail("cloud upload reached agent-browser"),
        )

        out = json.loads(browser_tool.browser_upload("#cv", [str(cv)]))

        assert out["success"] is False
        assert "cloud" in out["error"].lower()
        assert "not supported" in out["error"].lower()
        assert browser_tool.check_browser_upload_requirements() is False

    def test_remote_cdp_fails_closed_but_loopback_cdp_is_supported(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            browser_tool,
            "_get_cdp_override_raw",
            lambda: "wss://remote-browser.example/devtools/browser/123",
        )

        error = browser_tool._browser_upload_backend_error()
        assert error is not None
        assert "remote cdp" in error.lower()

        monkeypatch.setattr(
            browser_tool,
            "_get_cdp_override_raw",
            lambda: "ws://127.0.0.1:9222/devtools/browser/123",
        )

        assert browser_tool._browser_upload_backend_error() is None

    def test_backend_failure_is_returned_without_local_paths(self, monkeypatch, tmp_path):
        root = tmp_path / "staging"
        root.mkdir()
        cv = root / "cv.pdf"
        cv.write_bytes(b"pdf")
        _allow_root(monkeypatch, root)
        monkeypatch.setattr(
            browser_tool,
            "_run_browser_command",
            lambda *_args, **_kwargs: {"success": False, "error": "selector did not match"},
        )

        out = json.loads(browser_tool.browser_upload("#missing", [str(cv)], task_id="apply-1"))

        assert out == {"success": False, "error": "selector did not match"}
        assert str(root) not in json.dumps(out)


@pytest.mark.integration
@pytest.mark.live_system_guard_bypass
def test_real_agent_browser_uploads_to_hidden_file_input(monkeypatch, tmp_path):
    """Exercise browser_upload through the real agent-browser subprocess and Chrome."""
    if not browser_tool._chromium_installed():
        pytest.skip("Chromium is not installed")
    try:
        browser_tool._find_agent_browser()
    except FileNotFoundError:
        pytest.skip("agent-browser is not installed")

    html = tmp_path / "index.html"
    html.write_text(
        '<!doctype html><input id="hidden-upload" type="file" style="display:none">'
        '<script>window.uploadChanges=0; document.querySelector("#hidden-upload")'
        '.addEventListener("change",()=>window.uploadChanges++);</script>',
        encoding="utf-8",
    )
    payload = tmp_path / "cv smoke test.pdf"
    payload.write_bytes(b"%PDF-1.4\n% upload integration test\n")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    def handler(*args, **kwargs):
        return QuietHandler(*args, directory=str(tmp_path), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    task_id = f"upload-integration-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(browser_tool, "_get_upload_allowed_roots", lambda: [tmp_path.resolve()])
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(browser_tool, "_get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: True)

    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
        navigate = json.loads(browser_tool.browser_navigate(url, task_id=task_id))
        assert navigate["success"] is True, navigate

        missing = json.loads(browser_tool.browser_upload(
            "#does-not-exist", [str(payload)], task_id=task_id,
        ))
        assert missing["success"] is False, missing
        assert str(payload.resolve()) not in json.dumps(missing)

        upload = json.loads(browser_tool.browser_upload(
            "#hidden-upload", [str(payload)], task_id=task_id,
        ))
        verify = json.loads(browser_tool.browser_console(
            expression=(
                'JSON.stringify({name:document.querySelector("#hidden-upload").files[0]?.name,'
                'size:document.querySelector("#hidden-upload").files[0]?.size,'
                'changes:window.uploadChanges})'
            ),
            task_id=task_id,
        ))

        assert upload == {
            "success": True,
            "element": "#hidden-upload",
            "uploaded": [payload.name],
        }
        assert verify["result"] == {
            "name": payload.name,
            "size": payload.stat().st_size,
            "changes": 1,
        }
    finally:
        browser_tool.cleanup_browser(task_id)
        server.shutdown()
        server.server_close()
