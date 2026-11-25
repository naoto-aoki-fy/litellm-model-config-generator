# LiteLLM Proxy Config Generator

Generate a LiteLLM Proxy configuration from the models advertised by OpenAI-compatible APIs. The generator requires Python 3.9 or later and PyYAML.

Save at least one endpoint in a JSON or YAML file. The generator reads `.yaml` and `.yml` extensions as YAML regardless of case and reads all other filenames as JSON. Give each endpoint a unique name and replace the example keys with your API keys.

```json
{
  "endpoints": [
    {"name": "east", "url": "https://east.example.com/v1", "api_key": "key-east"},
    {"name": "west", "url": "https://west.example.com", "api_key": "key-west"}
  ]
}
```

The same endpoints in `endpoints.yaml` use the same field names. Quote YAML values that could be interpreted as numbers, booleans, or dates to keep them as strings.

```yaml
endpoints:
  - name: east
    url: https://east.example.com/v1
    api_key: "key-east"
  - name: west
    url: https://west.example.com
    api_key: "key-west"
```

Install the dependency and run the generator with either input file.

```sh
python3 -m pip install -r requirements.txt
python3 generate_config.py endpoints.json
python3 generate_config.py endpoints.yaml
litellm --config config.yaml
```

Use `-o PATH` to write to a different location. Choose a destination separate from the input file. The generator adds `/v1` where needed and prefixes shared model IDs with the endpoint name, for example `east/shared-model`. The generated file contains API keys.

The `litellm` command requires the LiteLLM Proxy CLI. See the [official LiteLLM setup and configuration examples](https://docs.litellm.ai/docs/).
