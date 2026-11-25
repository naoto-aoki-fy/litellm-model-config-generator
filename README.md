# LiteLLM Proxy Config Generator

Generate a LiteLLM Proxy configuration from the models advertised by OpenAI-compatible APIs. The generator requires Python 3.9 or later and PyYAML.

Save at least one endpoint in a JSON or YAML file. The generator reads `.yaml` and `.yml` extensions as YAML regardless of case and reads all other filenames as JSON. Give each endpoint a unique name and replace the example keys with your API keys.

Use the optional `model_defaults` mapping in each endpoint to apply settings to all models it exposes. Omit it or use `{}` to add no defaults for that endpoint. To migrate an existing input, move its top-level `model_defaults` into each endpoint that should use those settings. The generator rejects a top-level `model_defaults`, including `{}`.

Use the optional `model_overrides` mapping within an endpoint to apply settings to individual models. Each key must be the exact string ID returned by that endpoint's `/models` response, and each value must be a mapping of settings. The example applies the override only to `deepseek-ai/DeepSeek-V4-Flash` from `riken`. Omit `model_overrides` or use `{}` to leave an endpoint without overrides. The generator skips IDs that the endpoint did not return.

```json
{
  "endpoints": [
    {
      "name": "riken",
      "url": "https://api.class2.llm.ai.r-ccs.riken.jp/v1/",
      "api_key": "sk-xxxx",
      "model_defaults": {
        "model_info": {
          "supports_reasoning": false
        }
      },
      "model_overrides": {
        "deepseek-ai/DeepSeek-V4-Flash": {
          "litellm_params": {
            "use_chat_completions_api": true
          },
          "model_info": {
            "supports_reasoning": true,
            "reasoning_effort_levels": ["none", "low", "medium", "high"]
          }
        }
      }
    },
    {
      "name": "west",
      "url": "https://west.example.com",
      "api_key": "key-west",
      "model_defaults": {
        "model_info": {
          "supports_reasoning": false
        }
      }
    }
  ]
}
```

The same configuration in `endpoints.yaml` uses the same field names. Quote YAML values that could be interpreted as numbers, booleans, or dates when you want to keep them as strings.

```yaml
endpoints:
  - name: riken
    url: https://api.class2.llm.ai.r-ccs.riken.jp/v1/
    api_key: "sk-xxxx"
    model_defaults:
      model_info:
        supports_reasoning: false
    model_overrides:
      deepseek-ai/DeepSeek-V4-Flash:
        litellm_params:
          use_chat_completions_api: true
        model_info:
          supports_reasoning: true
          reasoning_effort_levels: [none, low, medium, high]
  - name: west
    url: https://west.example.com
    api_key: "key-west"
    model_defaults:
      model_info:
        supports_reasoning: false
```

For each model, the generator starts with its endpoint's `model_defaults`, overlays generated settings such as the public model name, connection parameters, and any matching `reasoning_templates` entry, then applies a matching `model_overrides` entry. When both values for a key are mappings, the generator merges them recursively. In all other cases, the later value replaces the earlier one, including whole lists. Adding `use_chat_completions_api` under `litellm_params` preserves `model`, `api_base`, and `api_key`. An override can also supply a replacement for any of those keys.

The generator validates `model_defaults` and `model_overrides` before contacting any endpoint. If an endpoint's `model_defaults` is invalid, the error identifies the endpoint by its position in the input. Invalid entries leave any existing output file intact, and error messages omit the input values.

Install the dependency and run the generator with either input file.

```sh
python3 -m pip install -r requirements.txt
python3 generate_config.py endpoints.json
python3 generate_config.py endpoints.yaml
litellm --config config.yaml
```

Use `-o PATH` to write to a different location. Choose a destination separate from the input file. The generator adds `/v1` where needed and prefixes shared model IDs with the endpoint name, for example `east/shared-model`. The generated file contains API keys.

The `litellm` command requires the LiteLLM Proxy CLI. See the [official LiteLLM setup and configuration examples](https://docs.litellm.ai/docs/).

## Per-model reasoning templates

Add `reasoning_templates` to any endpoint in the generator input. Use the same
mapping in JSON. The keys are exact model IDs from that endpoint's `/v1/models`
response, including when multiple endpoints advertise the same ID.

```yaml
endpoints:
  - name: east
    url: https://east.example.com/v1
    api_key: "key-east"
    reasoning_templates:
      "model-a": enable_thinking
      "model-b": thinking_mode
      "model-c": passthrough
```

- `enable_thinking` sets `chat_template_kwargs.enable_thinking` to a boolean.
- `thinking_mode` sets `chat_template_kwargs.thinking_mode` to `"enabled"` or `"disabled"`.
- `passthrough` records the choice in `model_info.reasoning_template` and leaves
  the request to LiteLLM's normal forwarding. Unlisted models also use normal
  forwarding.

For the first two types, an omitted effort, JSON `null`, or `"none"` disables
thinking and the hook omits the effort key from the upstream body. Other effort
values enable thinking and are forwarded as supplied. Chat Completions uses
`reasoning_effort`. Native Responses uses `reasoning.effort` and requires an
upstream that accepts `chat_template_kwargs` in the body. With
`use_chat_completions_api: true`, Responses requests instead send
`reasoning_effort` to upstream Chat Completions.

Use a LiteLLM Proxy version that supports the per-attempt deployment hook
(verified with 1.111.0). Put `reasoning_transform.py` beside the generated YAML
in the Proxy's filesystem. For a container, mount both files in the same
directory. The generator registers the module when at least one deployment's
merged settings select `enable_thinking` or `thinking_mode`. The hook reads
`model_info.reasoning_template` from the deployment selected for each attempt.

After editing the endpoint mappings, regenerate the configuration and restart
the Proxy with the resulting file. For a locally launched Proxy, stop the
current process before running the `litellm` command below.

```sh
python3 generate_config.py input.yaml -o config.yaml
litellm --config config.yaml
```
