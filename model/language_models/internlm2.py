from typing import Any, Dict, List, Optional

import torch
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from model.language_models.juice import JuiceChatModel


class InternLM2(JuiceChatModel):
    model_name: str = "InternLM2"
    tokenizer: object
    model: object
    device: str = "auto"
    model_id = "/home/user/ygz/base_model/Shanghai_AI_Laboratory/internlm2_5-7b-chat"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)
        tools = kwargs.get("tools", [])
        chat_history = process_input(message_dicts, tools)
        template = self.tokenizer.apply_chat_template(chat_history,
                                                      add_generation_prompt=True,
                                                      tokenize=False
                                                      )

        inputs = self.tokenizer(
            template,
            return_tensors="pt",
        ).to("cuda")

        gen_kwargs = self.gen_kwargs
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        function_call = process_response(response)
        tool_calls = None
        if isinstance(function_call, dict):
            finish_reason = "tool_calls"
            tool_calls = [
                {
                    "id": 1,
                    "function": function_call,
                    "type": "tool_use"
                }
            ]
        message = {
            "role": "assistant",
            "content": None if tool_calls else response,
            "function_call": None,
            "tool_calls": tool_calls
        }

        return self._create_chat_result(message)


def process_input(messages: List[Dict], tools: List[Dict]):
    if len(tools) == 0:
        return messages

    if messages[0]["role"] == "system":
        system_prompt = [{
            "role": "system",
            "content": messages[0]["content"],
            "tools": tools
        }]
        chat_history = system_prompt + messages[1:]
    else:
        system_prompt = [{
            "role": "system",
            "content": "",
            "tools": tools
        }]
        chat_history = system_prompt + messages

    return chat_history


def process_response(response: str):
    # lines = response.strip().split("\n")
    # if len(lines) >= 2 and lines[1].startswith("{"):
    #     function_name = lines[0].strip()
    #     arguments = "\n".join(lines[1:]).strip()
    #
    #     try:
    #         arguments_json = json.loads(arguments)
    #         is_tool_call = True
    #     except json.JSONDecodeError:
    #         is_tool_call = False
    #     if is_tool_call:
    #         content = {
    #             "name": function_name,
    #             "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
    #                                     ensure_ascii=False)
    #         }
    #         return content

    return response.strip()
