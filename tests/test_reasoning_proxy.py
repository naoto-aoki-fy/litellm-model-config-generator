"""Exercise the generated callback through an installed LiteLLM Proxy."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROXY = Path(sys.executable).with_name("litellm")
MISSING = object()
CHAT = "/v1/chat/completions"
RESPONSES = "/v1/responses"


class MockUpstream(BaseHTTPRequestHandler):
    def send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        self.send_body(status, "application/json", json.dumps(payload).encode())

    def do_GET(self):
        self.send_json(200, {"data": [
            {"id": model_id} for model_id in ("shared", "passthrough", "unlisted")
        ]})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, data))
        prompt = data["input"] if self.path.endswith(RESPONSES) else data["messages"][-1]["content"]
        if self.path.startswith("/east/") and prompt == "fallback":
            self.send_json(500, {"error": {"message": "Try the fallback"}})
            return

        if self.path.endswith(RESPONSES):
            response = {"id": "resp_mock", "object": "response", "created_at": 1,
                        "model": data["model"], "status": "completed", "output": []}
            completed = {"type": "response.completed", "response": response}
            stream = f"event: response.completed\ndata: {json.dumps(completed)}\n\n"
        else:
            base = {"id": "chatcmpl_mock", "created": 1, "model": data["model"]}
            message = {"role": "assistant", "content": "ok"}
            response = dict(base, object="chat.completion", choices=[
                {"index": 0, "message": message, "finish_reason": "stop"}
            ])
            chunk = dict(base, object="chat.completion.chunk", choices=[
                {"index": 0, "delta": message, "finish_reason": "stop"}
            ])
            stream = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"

        if data.get("stream"):
            self.send_body(200, "text/event-stream", stream.encode())
        else:
            self.send_json(200, response)

    def log_message(self, *args):
        pass


@unittest.skipUnless(PROXY.is_file(), "run with the Python environment that contains LiteLLM Proxy")
class ReasoningProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        work = Path(directory.name) / "config"
        work.mkdir()
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstream)
        cls.upstream = server
        server.requests = []
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        cls.addClassCleanup(server.server_close)
        cls.addClassCleanup(thread.join)
        cls.addClassCleanup(server.shutdown)

        base = f"http://127.0.0.1:{server.server_port}"
        document = {"endpoints": [
            {"name": name, "url": f"{base}/{name}", "api_key": "mock-api-key",
             "reasoning_templates": {"shared": template, "passthrough": "passthrough"}}
            for name, template in (("east", "enable_thinking"), ("west", "thinking_mode"))
        ]}
        (work / "input.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "generate_config.py"), "input.yaml"],
                       cwd=work, check=True, timeout=10)
        config_path = work / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for deployment in config["model_list"]:
            alias = deployment["model_name"]
            if alias.startswith("west/"):
                deployment["litellm_params"]["use_chat_completions_api"] = True
            if not alias.endswith("/shared"):
                # Let LiteLLM's usual forwarding accept "none" for the control models.
                deployment.setdefault("model_info", {}).update(
                    supports_reasoning=True, reasoning_effort_levels=["none", "high"],
                )
        config["router_settings"] = {
            "num_retries": 1,
            "fallbacks": [{"east/shared": ["west/shared"]}],
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        shutil.copyfile(ROOT / "reasoning_transform.py", work / "reasoning_transform.py")

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        cls.proxy_base = f"http://127.0.0.1:{port}"
        cls.key = "sk-" + secrets.token_hex(32)
        log_path = work / "proxy.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(PROXY), "--host", "127.0.0.1", "--port", str(port),
                 "--config", str(config_path), "--num_workers", "1"],
                cwd=work.parent, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True", LITELLM_MASTER_KEY=cls.key),
            )
        cls.addClassCleanup(cls.stop_proxy, process)
        deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(cls.proxy_base + "/health/liveliness", timeout=1):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("LiteLLM Proxy failed to start:\n" + log_path.read_text(encoding="utf-8"))

    @staticmethod
    def stop_proxy(process):
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def send(self, path, model, effort=MISSING, *, summary=False, stream=False, prompt="hello", **fields):
        data = dict(model=model, **fields)
        if path == CHAT:
            data["messages"] = [{"role": "user", "content": prompt}]
            if effort is not MISSING:
                data["reasoning_effort"] = effort
        else:
            data["input"] = prompt
            reasoning = {"summary": "auto"} if summary else {}
            if effort is not MISSING:
                reasoning["effort"] = effort
            if reasoning:
                data["reasoning"] = reasoning
        if stream:
            data["stream"] = True
        start = len(self.upstream.requests)
        request = Request(self.proxy_base + path, data=json.dumps(data).encode(), headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json",
        })
        with urlopen(request, timeout=12) as response:
            result = response.read()
        return result, self.upstream.requests[start:]

    def test_effort_and_template_keys_on_all_forwarding_paths(self):
        for path, endpoint, upstream_path, key, off, on in (
            (CHAT, "east", CHAT, "enable_thinking", False, True),
            (RESPONSES, "east", RESPONSES, "enable_thinking", False, True),
            (RESPONSES, "west", CHAT, "thinking_mode", "disabled", "enabled"),
        ):
            for effort in (MISSING, None, "none", "high", "custom", 0, False, {"custom": 2}):
                with self.subTest(path=path, endpoint=endpoint, effort=effort):
                    enabled = effort not in (MISSING, None, "none")
                    _, [(received_path, body)] = self.send(
                        path, f"{endpoint}/shared", effort, summary=True,
                        chat_template_kwargs={key: "overwritten", "other": {"keep": True}},
                        extra_body={"chat_template_kwargs": {"also_keep": 3}},
                    )
                    self.assertEqual(received_path, f"/{endpoint}{upstream_path}")
                    self.assertEqual(body["chat_template_kwargs"], {
                        key: on if enabled else off, "other": {"keep": True}, "also_keep": 3,
                    })
                    if upstream_path == RESPONSES:
                        expected = {"summary": "auto"}
                        if enabled:
                            expected["effort"] = effort
                        self.assertEqual(body["reasoning"], expected)
                        self.assertNotIn("reasoning_effort", body)
                    elif enabled:
                        self.assertEqual(body["reasoning_effort"], effort)
                    else:
                        self.assertNotIn("reasoning_effort", body)

    def test_empty_native_reasoning_object_is_omitted(self):
        _, [(_, body)] = self.send(RESPONSES, "east/shared", "none")
        self.assertNotIn("reasoning", body)

    def test_passthrough_and_unlisted_keep_explicit_none(self):
        for path, endpoint, upstream_path in (
            (CHAT, "east", CHAT), (RESPONSES, "east", RESPONSES), (RESPONSES, "west", CHAT)
        ):
            for model in ("passthrough", "unlisted"):
                with self.subTest(path=path, endpoint=endpoint, model=model):
                    template = {"enable_thinking": "as sent", "other": True}
                    _, [(received_path, body)] = self.send(
                        path, f"{endpoint}/{model}", "none", summary=True,
                        extra_body={"chat_template_kwargs": template},
                    )
                    self.assertEqual(received_path, f"/{endpoint}{upstream_path}")
                    self.assertEqual(body["chat_template_kwargs"], template)
                    if upstream_path == RESPONSES:
                        self.assertEqual(body["reasoning"], {"effort": "none", "summary": "auto"})
                    else:
                        self.assertEqual(body["reasoning_effort"], "none")

    def test_streaming_requests_use_the_same_transform(self):
        for path, endpoint, upstream_path, template in (
            (CHAT, "east", CHAT, {"enable_thinking": True}),
            (RESPONSES, "east", RESPONSES, {"enable_thinking": True}),
            (RESPONSES, "west", CHAT, {"thinking_mode": "enabled"}),
        ):
            with self.subTest(path=path, endpoint=endpoint):
                result, [(received_path, body)] = self.send(path, f"{endpoint}/shared", "high", stream=True)
                if path == CHAT:
                    self.assertIn(b"chat.completion.chunk", result)
                    self.assertIn(b"data: [DONE]", result)
                else:
                    self.assertIn(b"response.completed", result)
                self.assertEqual(received_path, f"/{endpoint}{upstream_path}")
                self.assertIs(body["stream"], True)
                self.assertEqual(body["chat_template_kwargs"], template)
                if upstream_path == RESPONSES:
                    self.assertEqual(body["reasoning"], {"effort": "high"})
                else:
                    self.assertEqual(body["reasoning_effort"], "high")

    def test_retry_and_fallback_recompute_template_for_the_selected_deployment(self):
        for path in (CHAT, RESPONSES):
            with self.subTest(path=path):
                _, attempts = self.send(
                    path, "east/shared", "high", prompt="fallback", chat_template_kwargs={"other": 42}
                )
                self.assertEqual([path for path, _ in attempts], ["/east" + path] * 2 + ["/west" + CHAT])
                for _, body in attempts[:-1]:
                    self.assertEqual(body["chat_template_kwargs"], {"other": 42, "enable_thinking": True})
                fallback = attempts[-1][1]
                self.assertEqual(fallback["chat_template_kwargs"], {"other": 42, "thinking_mode": "enabled"})
                self.assertEqual(fallback["reasoning_effort"], "high")
