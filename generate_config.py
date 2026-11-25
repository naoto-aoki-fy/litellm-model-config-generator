#!/usr/bin/env python3
"""Generate a LiteLLM Proxy configuration from OpenAI-compatible model lists."""

import argparse
from collections import Counter
from dataclasses import dataclass
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml


class GenerationError(Exception):
    """An error that can be reported without exposing endpoint data."""


@dataclass
class Endpoint:
    name: str
    api_base: str
    api_key: str


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep the Authorization header confined to the configured URL.
        return None


def reject_json_constant(value):
    raise ValueError("Invalid JSON constant")


def normalize_url(url):
    if re.search(r"[\x00-\x20\x7f]", url) or "?" in url or "#" in url:
        raise ValueError("Invalid base URL")
    parts = urlsplit(url)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
    ):
        raise ValueError("Invalid base URL")
    url = url.rstrip("/")
    return url if parts.path.rstrip("/").endswith("/v1") else url + "/v1"


def read_endpoints(path):
    is_yaml = path.suffix.lower() in (".yaml", ".yml")
    input_format = "YAML" if is_yaml else "JSON"
    load_error = f"Could not read the endpoints {input_format} file."
    try:
        with path.open(encoding="utf-8") as source:
            if is_yaml:
                try:
                    document = yaml.safe_load(source)
                except Exception:
                    # PyYAML can include secret scalar values in built-in exceptions.
                    raise GenerationError(load_error) from None
            else:
                document = json.load(source, parse_constant=reject_json_constant)
    except (OSError, ValueError, RecursionError):
        raise GenerationError(load_error) from None

    if not isinstance(document, dict) or not isinstance(document.get("endpoints"), list):
        raise GenerationError("The input must contain an endpoints array.")
    if not document["endpoints"]:
        raise GenerationError("At least one connection destination is required.")

    endpoints = []
    names = set()
    for position, item in enumerate(document["endpoints"], start=1):
        fields = ("name", "url", "api_key")
        if not isinstance(item, dict) or not all(
            isinstance(item.get(field), str) for field in fields
        ):
            raise GenerationError(
                f"Endpoint {position} must have string name, url, and api_key fields."
            )
        name, url, api_key = (item[field] for field in fields)
        if not name or name in names:
            raise GenerationError(f"Endpoint {position} must have a nonempty, unique name.")
        if re.search(r"[\x00-\x08\x0a-\x1f\x7f]", api_key):
            raise GenerationError(f"Endpoint {position} has an invalid API key header value.")
        try:
            api_base = normalize_url(url)
        except ValueError:
            raise GenerationError(f"Endpoint {position} has an invalid HTTP(S) base URL.") from None
        endpoints.append(Endpoint(name, api_base, api_key))
        names.add(name)
    return endpoints


def validate_output_path(input_path, output_path):
    try:
        same_file = output_path.samefile(input_path)
    except FileNotFoundError:
        return
    except OSError:
        raise GenerationError("Could not compare the input and output files.") from None
    if same_file:
        raise GenerationError("The input and output refer to the same file.")


def fetch_model_ids(opener, endpoint, position):
    try:
        request = Request(
            endpoint.api_base + "/models",
            headers={"Authorization": "Bearer " + endpoint.api_key},
            method="GET",
        )
        with opener.open(request, timeout=30) as response:
            body = response.read()
    except HTTPError as error:
        error.close()
        raise GenerationError(f"Endpoint {position} returned HTTP {error.code}.") from None
    except (OSError, HTTPException, ValueError):
        raise GenerationError(f"Could not fetch models from endpoint {position}.") from None

    try:
        document = json.loads(body, parse_constant=reject_json_constant)
    except (ValueError, RecursionError):
        raise GenerationError(f"Endpoint {position} returned invalid JSON.") from None

    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, list) or not all(
        isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        for item in data
    ):
        raise GenerationError(f"Endpoint {position} returned an invalid model list.")
    return list(dict.fromkeys(item["id"] for item in data))


def build_config(endpoints):
    opener = build_opener(NoRedirects())
    endpoint_models = [
        (endpoint, fetch_model_ids(opener, endpoint, position))
        for position, endpoint in enumerate(endpoints, start=1)
    ]
    counts = Counter(model_id for _, model_ids in endpoint_models for model_id in model_ids)
    entries = []
    public_names = set()
    for endpoint, model_ids in endpoint_models:
        for model_id in model_ids:
            public_name = model_id if counts[model_id] == 1 else f"{endpoint.name}/{model_id}"
            if public_name in public_names:
                raise GenerationError("Generated public model names conflict.")
            public_names.add(public_name)
            entries.append({
                "model_name": public_name,
                "litellm_params": {
                    "model": f"openai/{model_id}",
                    "api_base": endpoint.api_base,
                    "api_key": endpoint.api_key,
                },
            })
    return {"model_list": entries}


class _QuotedYamlString(str):
    """A string that PyYAML should emit with double quotes."""


def _represent_quoted_yaml_string(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style='"')


yaml.SafeDumper.add_representer(_QuotedYamlString, _represent_quoted_yaml_string)


def _quote_yaml_line_separators(value):
    # PyYAML can fold literal NEL characters to spaces in multiline scalars.
    # Double quotes let PyYAML escape the Unicode line separators.
    if isinstance(value, str) and any(separator in value for separator in "\x85\u2028\u2029"):
        return _QuotedYamlString(value)
    if isinstance(value, list):
        return [_quote_yaml_line_separators(item) for item in value]
    if isinstance(value, dict):
        return {
            _quote_yaml_line_separators(key): _quote_yaml_line_separators(item)
            for key, item in value.items()
        }
    return value


def write_config(path, config):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary_path = Path(output.name)
            yaml.safe_dump(
                _quote_yaml_line_separators(config),
                output,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "endpoint definitions in YAML (.yaml or .yml, case-insensitive) "
            "or JSON (all other filenames)"
        ),
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("config.yaml"),
        help="configuration destination (default: config.yaml)",
    )
    args = parser.parse_args()
    try:
        endpoints = read_endpoints(args.input)
        validate_output_path(args.input, args.output)
        config = build_config(endpoints)
    except GenerationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    try:
        write_config(args.output, config)
    except (OSError, ValueError):
        print("Error: Could not write the configuration.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
