# LiteLLM Proxy Config Generator

Generate a LiteLLM Proxy configuration from the models advertised by OpenAI-compatible APIs. The generator requires Python 3.9 or later and PyYAML.

Save at least one endpoint in a JSON or YAML file. The generator reads `.yaml` and `.yml` extensions as YAML regardless of case and reads all other filenames as JSON. Give each endpoint a unique name and replace the example keys with your API keys.

Use the optional `model_defaults` mapping alongside `endpoints` to apply common settings to every generated `model_list` entry. Omit it or use `{}` to generate entries without common settings.

```json
{
  "endpoints": [
    {"name": "east", "url": "https://east.example.com/v1", "api_key": "key-east"},
    {"name": "west", "url": "https://west.example.com", "api_key": "key-west"}
  ],
  "model_defaults": {
    "litellm_params": {
      "use_chat_completions_api": true
    },
    "model_info": {
      "supports_reasoning": true,
      "reasoning_effort_levels": ["none", "low", "medium", "high"]
    }
  }
}
```

The same configuration in `endpoints.yaml` uses the same field names. Quote YAML values that could be interpreted as numbers, booleans, or dates when you want to keep them as strings.

```yaml
endpoints:
  - name: east
    url: https://east.example.com/v1
    api_key: "key-east"
  - name: west
    url: https://west.example.com
    api_key: "key-west"
model_defaults:
  litellm_params:
    use_chat_completions_api: true
  model_info:
    supports_reasoning: true
    reasoning_effort_levels: [none, low, medium, high]
```

The generator recursively merges mappings and copies keys that appear on just one side. If a key appears in both the generated entry and `model_defaults`, both values must be mappings. For example, setting `litellm_params.model` in `model_defaults` causes a conflict even when its value matches the generated value. The error identifies the conflicting key path, omits the values, and leaves any existing output file intact.

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
directory. The generator registers the module when at least one discovered
deployment uses `enable_thinking` or `thinking_mode`. The hook reads
`model_info.reasoning_template` from the deployment selected for each attempt.

After editing the endpoint mappings, regenerate the configuration and restart
the Proxy with the resulting file. For a locally launched Proxy, stop the
current process before running the `litellm` command below.

```sh
python3 generate_config.py input.yaml -o config.yaml
litellm --config config.yaml
```
