"""Translate reasoning controls for the deployment selected by LiteLLM Proxy."""

from litellm.integrations.custom_logger import CustomLogger


def _mapping(value):
    return value if isinstance(value, dict) else {}


class ReasoningTransform(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        template = _mapping(kwargs.get("model_info")).get("reasoning_template")
        if template not in ("enable_thinking", "thinking_mode"):
            return kwargs
        if call_type not in ("acompletion", "aresponses"):
            return kwargs

        request = kwargs.copy()
        extra_body = _mapping(request.get("extra_body")).copy()
        template_kwargs = {
            **_mapping(extra_body.get("chat_template_kwargs")),
            **_mapping(request.pop("chat_template_kwargs", None)),
        }

        if call_type == "aresponses":
            reasoning = {
                **_mapping(extra_body.pop("reasoning", None)),
                **_mapping(request.pop("reasoning", None)),
            }
            effort = reasoning.pop("effort", None)
            enabled = effort is not None and effort != "none"
            if request.get("use_chat_completions_api") is True:
                # The bridge carries extra_body into the internal Chat call.
                # Keep Responses siblings available to LiteLLM's normal conversion.
                extra_body.pop("reasoning_effort", None)
                if enabled:
                    extra_body["reasoning_effort"] = effort
                if reasoning:
                    request["reasoning"] = reasoning
            else:
                if enabled:
                    reasoning["effort"] = effort
                if reasoning:
                    extra_body["reasoning"] = reasoning
        else:
            effort = request.pop("reasoning_effort", None)
            effort = extra_body.pop("reasoning_effort", effort)
            enabled = effort is not None and effort != "none"
            if enabled:
                extra_body["reasoning_effort"] = effort

        if template == "enable_thinking":
            template_kwargs[template] = enabled
        else:
            template_kwargs[template] = "enabled" if enabled else "disabled"
        # OpenAI-compatible transports merge extra_body into the final JSON.
        # This preserves upstream effort values without LiteLLM remapping them.
        extra_body["chat_template_kwargs"] = template_kwargs
        request["extra_body"] = extra_body
        return request


handler = ReasoningTransform()
