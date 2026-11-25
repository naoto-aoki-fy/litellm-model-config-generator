# LiteLLM Proxy Config Generator

Generate a LiteLLM Proxy configuration from the models advertised by OpenAI-compatible APIs. The generator requires Python 3.9 or later and PyYAML.

Save at least one endpoint in `endpoints.json`. Give each endpoint a unique name and replace the example keys with your API keys.

```json
{
  "endpoints": [
    {"name": "east", "url": "https://east.example.com/v1", "api_key": "key-east"},
    {"name": "west", "url": "https://west.example.com", "api_key": "key-west"}
  ]
}
```

```sh
python3 -m pip install -r requirements.txt
python3 generate_config.py endpoints.json
litellm --config config.yaml
```

Use `-o PATH` to write to a different location. Choose a destination separate from the input file. The generator adds `/v1` where needed and prefixes shared model IDs with the endpoint name, for example `east/shared-model`. The generated file contains API keys.

The `litellm` command requires the LiteLLM Proxy CLI. See the [official LiteLLM setup and configuration examples](https://docs.litellm.ai/docs/).
