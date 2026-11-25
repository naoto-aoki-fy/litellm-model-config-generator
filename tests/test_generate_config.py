import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from generate_config import normalize_url


SCRIPT = Path(__file__).resolve().parents[1] / "generate_config.py"
EAST_KEY = " east-key:# "
WEST_KEY = "west-key"
EAST_PATH = "/east/v1/models"
WEST_PATH = "/west/v1/models"


def model_list(*ids):
    return {"data": [{"id": model_id} for model_id in ids]}


class MockModelsAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        authorization = self.headers.get("Authorization")
        self.server.requests.append((self.path, authorization))
        expected_key = {EAST_PATH: EAST_KEY, WEST_PATH: WEST_KEY}.get(self.path)
        if expected_key is None:
            self.send_error(404)
            return
        if authorization != "Bearer " + expected_key:
            self.send_error(401)
            return
        status, payload = self.server.replies[self.path]
        if status == 302:
            self.send_response(status)
            self.send_header("Location", "/redirect-target")
            self.end_headers()
            return
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status, EAST_KEY if status == 401 else None)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class GenerateConfigTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MockModelsAPI)
        self.addCleanup(self.server.server_close)
        self.server.requests = []
        self.server.replies = {
            EAST_PATH: (200, model_list("shared", "east-only", "east-only")),
            WEST_PATH: (200, model_list("west-only", "shared", "shared")),
        }
        thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        base = f"http://127.0.0.1:{self.server.server_port}"
        self.east_base = base + "/east/v1"
        self.west_base = base + "/west/v1"
        self.endpoints = [
            {"name": "east", "url": self.east_base + "///", "api_key": EAST_KEY},
            {"name": "west", "url": base + "/west/", "api_key": WEST_KEY},
        ]

    def run_generator(self, *args):
        (self.directory / "endpoints.json").write_text(
            json.dumps({"endpoints": self.endpoints}), encoding="utf-8"
        )
        return subprocess.run(
            [sys.executable, str(SCRIPT), "endpoints.json", *args],
            cwd=self.directory,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def assert_preserved_on_failure(self, message):
        output_path = self.directory / "config.yaml"
        output_path.write_bytes(b"existing configuration\n")
        existing_inode = output_path.stat().st_ino
        result = self.run_generator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        for key in (EAST_KEY, WEST_KEY):
            self.assertNotIn(key.strip(), result.stderr + result.stdout)
        self.assertEqual(output_path.read_bytes(), b"existing configuration\n")
        self.assertEqual(output_path.stat().st_ino, existing_inode)
        self.assertEqual(
            {path.name for path in self.directory.iterdir()},
            {"config.yaml", "endpoints.json"},
        )

    def test_model_mapping_and_default_output(self):
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests, [
            (EAST_PATH, "Bearer " + EAST_KEY),
            (WEST_PATH, "Bearer " + WEST_KEY),
        ])
        output_path = self.directory / "config.yaml"
        entries = json.loads(output_path.read_text(encoding="utf-8"))["model_list"]
        self.assertEqual(
            [(item["model_name"], item["litellm_params"]) for item in entries],
            [
                (public_name, {"model": f"openai/{model_id}", "api_base": base, "api_key": key})
                for public_name, model_id, base, key in [
                    ("east/shared", "shared", self.east_base, EAST_KEY),
                    ("east-only", "east-only", self.east_base, EAST_KEY),
                    ("west-only", "west-only", self.west_base, WEST_KEY),
                    ("west/shared", "shared", self.west_base, WEST_KEY),
                ]
            ],
        )
        self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)

    def test_custom_output_replaces_existing_file_with_private_permissions(self):
        output_path = self.directory / "custom.yaml"
        output_path.write_text("old configuration", encoding="utf-8")
        output_path.chmod(0o644)
        result = self.run_generator("-o", output_path.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(output_path.read_text(encoding="utf-8"))["model_list"]), 4)
        self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
        self.assertFalse((self.directory / "config.yaml").exists())

    def test_authentication_error_preserves_output(self):
        self.server.replies[WEST_PATH] = (401, {"error": WEST_KEY})
        self.assert_preserved_on_failure("Endpoint 2 returned HTTP 401")

    def test_invalid_model_list_preserves_output(self):
        self.server.replies[WEST_PATH] = (200, {"data": [{"id": False}], "extra": WEST_KEY})
        self.assert_preserved_on_failure("Endpoint 2 returned an invalid model list")

    def test_public_name_conflict_preserves_output(self):
        self.server.replies[WEST_PATH] = (200, model_list("shared", "east/shared", WEST_KEY))
        self.assert_preserved_on_failure("Generated public model names conflict")

    def test_redirect_is_not_followed(self):
        self.server.replies[EAST_PATH] = (302, None)
        self.assert_preserved_on_failure("Endpoint 1 returned HTTP 302")
        self.assertEqual(self.server.requests, [(EAST_PATH, "Bearer " + EAST_KEY)])

    def test_v1_hostname_still_needs_a_path(self):
        self.assertEqual(normalize_url("http://v1/"), "http://v1/v1")
