import errno
from functools import partial
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

import yaml

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
        self.input_path = self.directory / "endpoints.json"
        self.write_input([
            {"name": "east", "url": self.east_base + "///", "api_key": EAST_KEY},
            {"name": "west", "url": base + "/west/", "api_key": WEST_KEY},
        ])

    def write_input(self, endpoints):
        self.input_path.write_text(
            json.dumps({"endpoints": endpoints}), encoding="utf-8"
        )

    def run_generator(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), self.input_path.name, *args],
            cwd=self.directory,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_reasoning_templates_match_exact_ids_per_endpoint_in_both_formats(self):
        endpoints = [
            {
                "name": "east", "url": self.east_base, "api_key": EAST_KEY,
                "model_defaults": {"model_info": {"supports_reasoning": True}},
                "reasoning_templates": {
                    "shared": "enable_thinking",
                    "east-only": "passthrough",
                    "west-only": "enable_thinking",
                    "SHARED": "thinking_mode",
                },
            },
            {
                "name": "west", "url": self.west_base, "api_key": WEST_KEY,
                "model_defaults": {"model_info": {"supports_reasoning": True}},
                "reasoning_templates": {"shared": "thinking_mode"},
            },
        ]
        document = {"endpoints": endpoints}
        for extension, dump in ((".json", json.dumps), (".yaml", yaml.safe_dump)):
            with self.subTest(extension=extension):
                self.input_path = self.input_path.with_suffix(extension)
                self.input_path.write_text(dump(document), encoding="utf-8")
                result = self.run_generator()
                self.assertEqual(result.returncode, 0, result.stderr)
                config = yaml.safe_load((self.directory / "config.yaml").read_text())
                self.assertEqual(config["litellm_settings"]["callbacks"], ["reasoning_transform.handler"])
                self.assertEqual({
                    entry["model_name"]: entry["model_info"] for entry in config["model_list"]
                }, {
                    "east/shared": {"reasoning_template": "enable_thinking", "supports_reasoning": True},
                    "east-only": {"reasoning_template": "passthrough", "supports_reasoning": True},
                    "west/shared": {"reasoning_template": "thinking_mode", "supports_reasoning": True},
                    "west-only": {"supports_reasoning": True},
                })

    def test_reasoning_templates_without_matching_transform_do_not_register_callback(self):
        self.write_input([{
            "name": "east", "url": self.east_base, "api_key": EAST_KEY,
            "reasoning_templates": {"shared": "passthrough", "unknown": "thinking_mode"},
        }])
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((self.directory / "config.yaml").read_text())
        self.assertNotIn("litellm_settings", config)
        shared, unlisted = config["model_list"]
        self.assertEqual(shared["model_info"], {"reasoning_template": "passthrough"})
        self.assertNotIn("model_info", unlisted)

    def test_invalid_reasoning_templates_preserve_output_before_discovery(self):
        invalid_cases = [
            (None, "must be a mapping"),
            ([], "must be a mapping"),
            ({42: "passthrough"}, "must have string model IDs"),
            ({"unknown": "enable-thinkng"}, "values must be"),
            ({"unknown": ["enable_thinking"]}, "values must be"),
            ({"unknown": True}, "values must be"),
            ({"unknown": None}, "values must be"),
        ]
        for extension, dump in ((".json", json.dumps), (".yaml", yaml.safe_dump)):
            self.input_path.unlink()
            self.input_path = self.input_path.with_suffix(extension)
            for mapping, error in invalid_cases:
                if extension == ".json" and isinstance(mapping, dict) and 42 in mapping:
                    continue  # JSON object keys are always strings.
                with self.subTest(extension=extension, mapping=mapping):
                    self.input_path.write_text(dump({"endpoints": [{
                        "name": "east", "url": self.east_base, "api_key": EAST_KEY,
                        "reasoning_templates": mapping,
                    }]}), encoding="utf-8")
                    self.assert_preserved_on_failure("reasoning_templates " + error)
        self.assertEqual(self.server.requests, [])

    def assert_preserved_on_failure(self, message):
        output_path = self.directory / "config.yaml"
        output_path.write_bytes(b"existing configuration\n")
        existing_inode = output_path.stat().st_ino
        result = self.run_generator()
        self.assertEqual(result.returncode, 1)
        self.assertIn(message, result.stderr)
        for key in (EAST_KEY, WEST_KEY):
            self.assertNotIn(key.strip(), result.stderr + result.stdout)
        self.assertEqual(output_path.read_bytes(), b"existing configuration\n")
        self.assertEqual(output_path.stat().st_ino, existing_inode)
        self.assertEqual(
            {path.name for path in self.directory.iterdir()},
            {output_path.name, self.input_path.name},
        )
        return result

    def assert_input_preserved(self, output_path):
        original = self.input_path.read_bytes()
        result = self.run_generator("-o", str(output_path))
        self.assertEqual(result.returncode, 1)
        self.assertIn("The input and output refer to the same file", result.stderr)
        self.assertEqual(self.input_path.read_bytes(), original)
        self.assertTrue(self.input_path.samefile(self.directory / output_path))
        self.assertEqual(self.server.requests, [])

    def create_link(self, create):
        try:
            create()
        except NotImplementedError as error:
            self.skipTest(f"Link creation is unavailable: {error}")
        except OSError as error:
            unsupported = (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP)
            # Windows can report unsupported filesystems or missing link privileges.
            if error.errno in unsupported or getattr(error, "winerror", None) in (1, 50, 1314):
                self.skipTest(f"Link creation is unavailable: {error}")
            raise

    def test_model_mapping_and_default_output(self):
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests, [
            (EAST_PATH, "Bearer " + EAST_KEY),
            (WEST_PATH, "Bearer " + WEST_KEY),
        ])
        output_path = self.directory / "config.yaml"
        contents = output_path.read_text(encoding="utf-8")
        self.assertTrue(contents.startswith("model_list:\n"))
        entries = yaml.safe_load(contents)["model_list"]
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

    def test_input_formats_generate_identical_configurations(self):
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        output_path = self.directory / "config.yaml"
        expected = output_path.read_bytes()
        json_input = self.input_path.read_text(encoding="utf-8")
        yaml_input = yaml.safe_dump(json.loads(json_input))

        for filename, contents in (
            ("endpoints.yaml", yaml_input),
            ("endpoints.yml", yaml_input),
            ("endpoints.YAML", yaml_input),
            ("endpoints.YmL", yaml_input),
            ("endpoints", json_input),
            ("endpoints.txt", json_input),
        ):
            with self.subTest(filename=filename):
                output_path.unlink(missing_ok=True)
                self.input_path = self.directory / filename
                self.input_path.write_text(contents, encoding="utf-8")
                result = self.run_generator()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output_path.read_bytes(), expected)

    def test_model_defaults_apply_only_to_models_from_their_endpoint_in_both_formats(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        east, west = document["endpoints"]
        east["model_defaults"] = {
            "z_custom_field": True,
            "litellm_params": {
                "use_chat_completions_api": True,
                "custom_parameter": {"options": [False, "custom"]},
            },
            "model_info": {
                "supports_reasoning": True,
                "reasoning_effort_levels": ["none", "low", "medium", "high"],
            },
            "custom_field": {"allocation": [3, 1, 2]},
        }
        west_defaults = {
            "litellm_params": {"use_chat_completions_api": False},
            "model_info": {"supports_reasoning": False},
            "custom_field": {"allocation": [4]},
        }
        output_path = self.directory / "config.yaml"
        expected_entries = [
            ("east/shared", "shared", east, self.east_base),
            ("east-only", "east-only", east, self.east_base),
            ("west-only", "west-only", west, self.west_base),
            ("west/shared", "shared", west, self.west_base),
        ]
        for extension, dump in (
            (".json", json.dumps),
            (".yaml", partial(yaml.safe_dump, sort_keys=False)),
        ):
            self.input_path = self.input_path.with_suffix(extension)
            for defaults_setting in (west_defaults, None, {}):
                with self.subTest(extension=extension, west_defaults=defaults_setting):
                    if defaults_setting is None:
                        west.pop("model_defaults", None)
                    else:
                        west["model_defaults"] = defaults_setting
                    self.input_path.write_text(dump(document), encoding="utf-8")
                    output_path.unlink(missing_ok=True)
                    result = self.run_generator()
                    self.assertEqual(result.returncode, 0, result.stderr)
                    entries = yaml.safe_load(output_path.read_text(encoding="utf-8"))["model_list"]
                    self.assertEqual(len(entries), len(expected_entries))
                    for entry, (name, model_id, endpoint, api_base) in zip(entries, expected_entries):
                        defaults = endpoint.get("model_defaults", {})
                        expected_keys = ["model_name", "litellm_params"]
                        if endpoint is east:
                            expected_keys.extend(["model_info", "z_custom_field", "custom_field"])
                        elif defaults:
                            expected_keys.extend(["model_info", "custom_field"])
                        self.assertEqual(list(entry), expected_keys, name)
                        self.assertEqual(entry, {
                            **defaults,
                            "model_name": name,
                            "litellm_params": {
                                **defaults.get("litellm_params", {}),
                                "model": f"openai/{model_id}",
                                "api_base": api_base,
                                "api_key": endpoint["api_key"],
                            },
                        })
                        if defaults:
                            self.assertIs(
                                entry["litellm_params"]["use_chat_completions_api"],
                                defaults["litellm_params"]["use_chat_completions_api"],
                            )
                            self.assertIs(
                                entry["model_info"]["supports_reasoning"],
                                defaults["model_info"]["supports_reasoning"],
                            )

    def test_model_overrides_match_exact_ids_per_endpoint_in_both_formats(self):
        model_id = "deepseek-ai/DeepSeek-V4-Flash"
        self.server.replies[EAST_PATH] = (200, model_list(model_id, "east-only"))
        self.server.replies[WEST_PATH] = (200, model_list(model_id, "west-only"))
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        riken, _ = document["endpoints"]
        riken["name"] = "riken"
        effort_levels = ["none", "low", "medium", "high"]
        riken["model_overrides"] = {
            model_id: {
                "z_custom_field": True,
                "litellm_params": {"use_chat_completions_api": True},
                "custom_field": {"allocation": [1]},
                "model_info": {
                    "supports_reasoning": True,
                    "reasoning_effort_levels": effort_levels,
                },
            },
            model_id.lower(): {"unexpected": "different case"},
            f"riken/{model_id}": {"unexpected": "public name"},
        }
        for extension, dump in (
            (".json", json.dumps),
            (".yaml", partial(yaml.safe_dump, sort_keys=False)),
        ):
            with self.subTest(extension=extension):
                self.input_path = self.input_path.with_suffix(extension)
                self.input_path.write_text(dump(document), encoding="utf-8")
                result = self.run_generator()
                self.assertEqual(result.returncode, 0, result.stderr)
                config = yaml.safe_load((self.directory / "config.yaml").read_text())
                entries = {entry["model_name"]: entry for entry in config["model_list"]}
                target = entries.pop(f"riken/{model_id}")
                self.assertEqual(list(target), [
                    "model_name", "litellm_params", "model_info", "z_custom_field", "custom_field",
                ])
                self.assertIs(target["z_custom_field"], True)
                self.assertEqual(target["custom_field"], {"allocation": [1]})
                self.assertIs(target["litellm_params"]["use_chat_completions_api"], True)
                self.assertIs(target["model_info"]["supports_reasoning"], True)
                self.assertEqual(target["model_info"]["reasoning_effort_levels"], effort_levels)
                self.assertEqual(target["litellm_params"], {
                    "model": f"openai/{model_id}",
                    "api_base": self.east_base,
                    "api_key": EAST_KEY,
                    "use_chat_completions_api": True,
                })
                self.assertNotIn("unexpected", target)
                self.assertEqual(set(entries), {"east-only", "west-only", f"west/{model_id}"})
                for entry in entries.values():
                    self.assertEqual(list(entry), ["model_name", "litellm_params"])
                    self.assertNotIn("use_chat_completions_api", entry["litellm_params"])
                    self.assertNotIn("unexpected", entry)

    def test_generated_settings_overwrite_model_defaults(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        east, _ = document["endpoints"]
        east["reasoning_templates"] = {"shared": "enable_thinking"}
        for params, extra in (
            ({"model": "default-model", "api_base": "default-base", "api_key": "default-key",
              "custom": "preserved"}, {"custom": "preserved"}),
            (["default-params"], {}),
        ):
            with self.subTest(params=params):
                east["model_defaults"] = {
                    "model_name": {"nested": "default-name"},
                    "litellm_params": params,
                    "model_info": {"reasoning_template": "passthrough", "supports_reasoning": True},
                }
                self.input_path.write_text(json.dumps(document), encoding="utf-8")
                result = self.run_generator()
                self.assertEqual(result.returncode, 0, result.stderr)
                config = yaml.safe_load((self.directory / "config.yaml").read_text())
                self.assertEqual(config["model_list"][0], {
                    "model_name": "east/shared",
                    "litellm_params": {
                        "model": "openai/shared", "api_base": self.east_base, "api_key": EAST_KEY,
                        **extra,
                    },
                    "model_info": {"reasoning_template": "enable_thinking", "supports_reasoning": True},
                })

    def test_model_overrides_overwrite_generated_and_nested_default_values(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        east, _ = document["endpoints"]
        east["model_defaults"] = {
            "custom": {
                "nested": {
                    "preserved": True, "levels": ["default", "removed"],
                    "object": {"key": "removed"}, "scalar": "replaced",
                },
            },
        }
        east["reasoning_templates"] = {"shared": "enable_thinking"}
        east["model_overrides"] = {
            "shared": {
                "model_name": "override-name",
                "litellm_params": {"model": "override-model", "api_key": "override-key"},
                "model_info": {"reasoning_template": "passthrough"},
                "custom": {"nested": {"levels": ["overridden"], "object": None, "scalar": {"key": 2}}},
            },
        }
        self.input_path.write_text(json.dumps(document), encoding="utf-8")
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((self.directory / "config.yaml").read_text())
        self.assertEqual(config["model_list"][0], {
            "model_name": "override-name",
            "litellm_params": {
                "model": "override-model", "api_base": self.east_base, "api_key": "override-key",
            },
            "model_info": {"reasoning_template": "passthrough"},
            "custom": {
                "nested": {
                    "preserved": True, "levels": ["overridden"], "object": None, "scalar": {"key": 2},
                },
            },
        })
        self.assertEqual(config["model_list"][1]["custom"], east["model_defaults"]["custom"])
        self.assertNotIn("litellm_settings", config)

    def test_empty_and_undiscovered_model_overrides_match_omitted_overrides(self):
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        output_path = self.directory / "config.yaml"
        expected = output_path.read_bytes()
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        for overrides in ({}, {"shared": {}}, {"west-only": {"custom": True}}):
            with self.subTest(overrides=overrides):
                document["endpoints"][0]["model_overrides"] = overrides
                self.input_path.write_text(json.dumps(document), encoding="utf-8")
                result = self.run_generator()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output_path.read_bytes(), expected)

    def test_invalid_model_overrides_preserve_output_before_discovery(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        self.input_path.unlink()
        invalid_cases = [
            (None, "must be a mapping"),
            (["private-setting"], "must be a mapping"),
            ("private-setting", "must be a mapping"),
            ({"private-id": "private-setting"}, "values must be mappings"),
            ({"private-id": ["private-setting"]}, "values must be mappings"),
            ({"private-id": None}, "values must be mappings"),
            ({42: {"custom": "private-setting"}}, "must have string model IDs"),
        ]
        for extension, dump in ((".json", json.dumps), (".yaml", yaml.safe_dump)):
            self.input_path = self.input_path.with_suffix(extension)
            for overrides, error in invalid_cases:
                if extension == ".json" and isinstance(overrides, dict) and 42 in overrides:
                    continue  # JSON object keys are always strings.
                with self.subTest(extension=extension, overrides=overrides):
                    document["endpoints"][1]["model_overrides"] = overrides
                    self.input_path.write_text(dump(document), encoding="utf-8")
                    message = f"Endpoint 2 model_overrides {error}."
                    result = self.assert_preserved_on_failure(message)
                    self.assertEqual(result.stderr, f"Error: {message}\n")
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(self.server.requests, [])
            self.input_path.unlink()

    def test_top_level_model_defaults_require_migration_before_discovery(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        for extension, dump in ((".json", json.dumps), (".yaml", yaml.safe_dump)):
            self.input_path.unlink()
            self.input_path = self.input_path.with_suffix(extension)
            for value in ({}, {"custom": "private-default"}, None):
                with self.subTest(extension=extension, defaults=value):
                    document["model_defaults"] = value
                    self.input_path.write_text(dump(document), encoding="utf-8")
                    message = "Move top-level model_defaults into each endpoint that should use them."
                    result = self.assert_preserved_on_failure(message)
                    self.assertEqual(result.stderr, f"Error: {message}\n")
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(self.server.requests, [])

    def test_invalid_endpoint_model_defaults_preserve_output_before_discovery(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        for extension, dump in ((".json", json.dumps), (".yaml", yaml.safe_dump)):
            self.input_path.unlink()
            self.input_path = self.input_path.with_suffix(extension)
            for value in (None, False, 0, "private-default", ["private-default"]):
                with self.subTest(extension=extension, value_type=type(value).__name__):
                    document["endpoints"][1]["model_defaults"] = value
                    self.input_path.write_text(dump(document), encoding="utf-8")
                    message = "Endpoint 2 model_defaults must be a mapping."
                    result = self.assert_preserved_on_failure(message)
                    self.assertEqual(result.stderr, f"Error: {message}\n")
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(self.server.requests, [])

    def test_yaml_syntax_error_preserves_output_and_hides_input(self):
        self.input_path.unlink()
        self.input_path = self.directory / "endpoints.yaml"
        self.input_path.write_text(
            f'endpoints:\n  - {{api_key: "{EAST_KEY}", name: ]}}\n',
            encoding="utf-8",
        )
        self.assert_preserved_on_failure("Could not read the endpoints YAML file.")
        self.assertEqual(self.server.requests, [])

    def test_invalid_yaml_scalar_tags_preserve_output_and_hide_keys(self):
        self.input_path.unlink()
        self.input_path = self.directory / "endpoints.yaml"
        for scalar in (f'!!bool "{EAST_KEY}"', f'!!timestamp "{EAST_KEY}"', '!!int ""'):
            with self.subTest(tag=scalar.split()[0]):
                self.input_path.write_text(
                    f"endpoints:\n  - name: east\n    url: {self.east_base}\n"
                    f"    api_key: {scalar}\n",
                    encoding="utf-8",
                )
                self.assert_preserved_on_failure("Could not read the endpoints YAML file.")
                self.assertEqual(self.server.requests, [])

    def test_yaml_required_field_type_mismatches_preserve_output_and_hide_keys(self):
        self.input_path.unlink()
        self.input_path = self.directory / "endpoints.yml"
        for field, value in (("name", True), ("url", None), ("api_key", [EAST_KEY])):
            with self.subTest(field=field):
                endpoint = {"name": "east", "url": self.east_base, "api_key": EAST_KEY}
                endpoint[field] = value
                self.input_path.write_text(
                    yaml.safe_dump({"endpoints": [endpoint]}), encoding="utf-8"
                )
                self.assert_preserved_on_failure(
                    "Endpoint 1 must have string name, url, and api_key fields."
                )
                self.assertEqual(self.server.requests, [])

    def test_custom_output_replaces_existing_file(self):
        output_path = self.directory / "custom.yaml"
        output_path.write_text("old configuration", encoding="utf-8")
        result = self.run_generator("-o", output_path.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(yaml.safe_load(output_path.read_text(encoding="utf-8"))["model_list"]), 4)
        self.assertFalse((self.directory / "config.yaml").exists())

    def test_unicode_model_id_survives_yaml_round_trip(self):
        model_id = "東京/😀\x85line\u2028separator\u2029end"
        self.server.replies[EAST_PATH] = (200, model_list(model_id))
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        contents = (self.directory / "config.yaml").read_text(encoding="utf-8")
        self.assertIn("東京/", contents)
        entry = yaml.safe_load(contents)["model_list"][0]
        self.assertEqual(entry["model_name"], model_id)
        self.assertEqual(entry["litellm_params"]["model"], f"openai/{model_id}")

    def test_output_cannot_be_the_input_path(self):
        self.assert_input_preserved(self.input_path.name)

    def test_output_cannot_be_an_absolute_alias_of_the_input(self):
        self.assert_input_preserved(self.input_path.absolute())

    def test_output_cannot_be_a_symbolic_link_to_the_input(self):
        output_path = self.directory / "symbolic-link.yaml"
        self.create_link(lambda: output_path.symlink_to(self.input_path))
        self.assert_input_preserved(output_path)

    def test_output_cannot_be_a_hard_link_to_the_input(self):
        output_path = self.directory / "hard-link.yaml"
        self.create_link(lambda: os.link(self.input_path, output_path))
        self.assert_input_preserved(output_path)

    def test_empty_endpoints_preserves_output(self):
        self.write_input([])
        self.assert_preserved_on_failure("At least one connection destination is required")
        self.assertEqual(self.server.requests, [])

    def test_empty_model_list_is_valid(self):
        self.write_input([{"name": "east", "url": self.east_base, "api_key": EAST_KEY}])
        self.server.replies[EAST_PATH] = (200, model_list())
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests, [(EAST_PATH, "Bearer " + EAST_KEY)])
        output_path = self.directory / "config.yaml"
        self.assertEqual(
            yaml.safe_load(output_path.read_text(encoding="utf-8")),
            {"model_list": []},
        )

    def test_exact_exclusion_is_endpoint_specific_and_precedes_duplicate_detection(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        document["endpoints"][0]["exclude_models"] = ["shared"]
        self.server.replies[EAST_PATH] = (200, model_list("east-only", "shared", "after-shared"))
        self.server.replies[WEST_PATH] = (200, model_list("west-only", "shared", "east/shared"))
        for extension, dump in ((".json", json.dumps), (".yaml", yaml.safe_dump)):
            with self.subTest(extension=extension):
                self.input_path = self.input_path.with_suffix(extension)
                self.input_path.write_text(dump(document), encoding="utf-8")
                result = self.run_generator()
                self.assertEqual(result.returncode, 0, result.stderr)
                config = yaml.safe_load((self.directory / "config.yaml").read_text())
                self.assertEqual([
                    (entry["model_name"], entry["litellm_params"]["api_base"])
                    for entry in config["model_list"]
                ], [
                    ("east-only", self.east_base),
                    ("after-shared", self.east_base),
                    ("west-only", self.west_base),
                    ("shared", self.west_base),
                    ("east/shared", self.west_base),
                ])

    def test_wildcards_match_whole_case_sensitive_original_ids(self):
        self.write_input([{
            "name": "east", "url": self.east_base, "api_key": EAST_KEY,
            "exclude_models": ["deepseek-ai/*", "*-preview", "chat-?", "embed-[abc]", "rank-[!abc]"],
        }])
        self.server.replies[EAST_PATH] = (200, model_list(
            "deepseek-ai/nested/model", "deepseek-ai/", "vendor/deepseek-ai/model",
            "vendor/model-preview", "vendor/model-preview-old", "vendor/model-Preview",
            "chat-1", "chat-10", "embed-b", "embed-d", "rank-x", "rank-c",
        ))
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((self.directory / "config.yaml").read_text())
        self.assertEqual([entry["model_name"] for entry in config["model_list"]], [
            "vendor/deepseek-ai/model", "vendor/model-preview-old", "vendor/model-Preview",
            "chat-10", "embed-d", "rank-c",
        ])

    def test_empty_and_unmatched_exclusions_match_omitted_exclusions(self):
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        output_path = self.directory / "config.yaml"
        expected = output_path.read_bytes()
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        for patterns in ([], ["missing-*", "east/shared", "openai/east-only"]):
            with self.subTest(patterns=patterns):
                document["endpoints"][0]["exclude_models"] = patterns
                self.input_path.write_text(json.dumps(document), encoding="utf-8")
                result = self.run_generator()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output_path.read_bytes(), expected)

    def test_excluded_models_skip_settings_and_callback_registration(self):
        self.write_input([{
            "name": "east", "url": self.east_base, "api_key": EAST_KEY,
            "exclude_models": ["*"],
            "model_defaults": {"model_info": {"reasoning_template": "enable_thinking"}},
            "model_overrides": {"shared": {"model_name": "revived-by-override"}},
        }, {
            "name": "west", "url": self.west_base, "api_key": WEST_KEY,
            "exclude_models": ["shared"],
            "reasoning_templates": {"shared": "thinking_mode"},
            "model_overrides": {"shared": {"model_name": "also-revived-by-override"}},
        }])
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((self.directory / "config.yaml").read_text())
        self.assertEqual(config, {"model_list": [{
            "model_name": "west-only",
            "litellm_params": {
                "model": "openai/west-only", "api_base": self.west_base, "api_key": WEST_KEY,
            },
        }]})

    def test_excluding_all_models_produces_an_empty_model_list(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        for endpoint in document["endpoints"]:
            endpoint["exclude_models"] = ["*"]
        self.input_path.write_text(json.dumps(document), encoding="utf-8")
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests, [
            (EAST_PATH, "Bearer " + EAST_KEY),
            (WEST_PATH, "Bearer " + WEST_KEY),
        ])
        self.assertEqual((self.directory / "config.yaml").read_text(), "model_list: []\n")

    def test_invalid_exclusions_preserve_output_before_discovery(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        self.input_path.unlink()
        invalid_cases = (
            (None, "must be an array"),
            ("private-pattern", "must be an array"),
            ({"private-pattern": True}, "must be an array"),
            (["private-pattern", {"private-value": True}], "must contain only nonempty strings"),
            (["private-pattern", False], "must contain only nonempty strings"),
            (["private-pattern", ""], "must contain only nonempty strings"),
        )
        for extension, dump in ((".json", json.dumps), (".yaml", yaml.safe_dump)):
            self.input_path = self.input_path.with_suffix(extension)
            for patterns, error in invalid_cases:
                with self.subTest(extension=extension, patterns=patterns):
                    document["endpoints"][1]["exclude_models"] = patterns
                    self.input_path.write_text(dump(document), encoding="utf-8")
                    message = f"Endpoint 2 exclude_models {error}."
                    result = self.assert_preserved_on_failure(message)
                    self.assertEqual(result.stderr, f"Error: {message}\n")
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(self.server.requests, [])
            self.input_path.unlink()

    def test_exclusions_do_not_bypass_model_list_validation(self):
        document = json.loads(self.input_path.read_text(encoding="utf-8"))
        document["endpoints"][1]["exclude_models"] = ["*"]
        self.input_path.write_text(json.dumps(document), encoding="utf-8")
        self.server.replies[WEST_PATH] = (200, model_list("shared", False))
        self.assert_preserved_on_failure("Endpoint 2 returned an invalid model list.")

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
