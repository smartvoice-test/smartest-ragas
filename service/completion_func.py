import os
import uuid
import json
import asyncio
import typing as t

import httpx
from jsonpath import jsonpath
from dotenv import load_dotenv, find_dotenv
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ragas.llms.base import BaseRagasLLM
from ragas.llms.prompt import PromptValue

_ = load_dotenv(find_dotenv())
_authorization = os.getenv("OpenAPIKey")


def get_prompt_json(prompt: PromptValue):
    messages = []
    for row in prompt.to_messages():
        if isinstance(row, SystemMessage):
            messages.append({"role": "system", "content": row.content})
        elif isinstance(row, HumanMessage):
            messages.append({"role": "user", "content": row.content})
        elif isinstance(row, AIMessage):
            messages.append({"role": "assistant", "content": row.content})
    return messages


class GatewayLLM(BaseRagasLLM):
    def __init__(
            self,
            model: str,
            address: str = "http://172.16.23.86:30385/v1/chat/completions",
            api_timeout: int = 60,
            max_tokens: int = 2048,
            do_sample: bool = True,
            top_p: float = 0.1,
            repetition_penalty: float = 1,
    ):
        self.address = address
        self.model = model
        self.max_tokens = max_tokens
        self.api_timeout = api_timeout
        self.do_sample = do_sample
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    @property
    def _headers(self):
        return {
            "Content-Type": "application/json; charset=utf-8",
            "ApiName": "robotGptLLMApi",
            "Model": self.model,
            "Authorization": _authorization
        }

    def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks: Callbacks = [],
    ) -> LLMResult:
        request_json = {
            "model": self.model,
            "temperature": temperature,
            "n": n,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "max_new_token": self.max_tokens,
            "api_timeout": self.api_timeout,
            "do_sample": self.do_sample,
            "stream": False,
            "stop": stop,
            "messages": get_prompt_json(prompt),
            "trace_id": f"RAGAS{uuid.uuid4().hex}@cloudminds-test.com.cn"
        }

        try:
            with httpx.Client(verify=False) as client:
                response = client.post(
                    self.address,
                    headers=self._headers,
                    json=request_json,
                    timeout=self.api_timeout,
                )
                response.raise_for_status()
                ret = response.json()
                return LLMResult(generations=[[Generation(text="".join(jsonpath(ret, "$.choices..message.content")))]])
        except httpx.HTTPStatusError as e:
            print(f"HTTP request failed with status {e.response.status_code}.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        return LLMResult(generations=[[Generation(text="")]])

    async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks: Callbacks = [],
    ) -> LLMResult:
        request_json = {
            "model": self.model,
            "temperature": temperature,
            "n": n,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "max_new_token": self.max_tokens,
            "api_timeout": self.api_timeout,
            "do_sample": self.do_sample,
            "stream": True,
            "stop": stop,
            "messages": get_prompt_json(prompt),
            "trace_id": f"RAGAS{uuid.uuid4().hex}@cloudminds-test.com.cn"
        }

        completions = []
        try:
            async with httpx.AsyncClient(verify=False) as client:
                async with client.stream(
                        "POST",
                        self.address,
                        headers=self._headers,
                        json=request_json,
                        timeout=self.api_timeout,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        chunk = chunk.strip() if isinstance(chunk, str) else chunk
                        if not chunk:
                            continue
                        data = json.loads(chunk.removeprefix("data:"))
                        for choice in data["choices"]:
                            if choice["finish_reason"] in ["stop"]:
                                return LLMResult(generations=[[Generation(text="".join(completions))]])
                            if "delta" in choice:
                                completions.append(choice["delta"]["content"])
                            elif "message" in choice:
                                completions.append(choice["message"]["content"])
        except httpx.HTTPStatusError as e:
            print(f"HTTP request failed with status {e.response.status_code}.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        return LLMResult(generations=[[Generation(text="".join(completions))]])


async def main():
    llm = GatewayLLM(
        # address="https://dataai.harix.iamidata.com/llm/api/ask",
        # model="zhipuai/glm-4",
        model="robotgpt-rc-trt-pcache",
    )
    p = PromptValue(prompt_str="你好")
    llm_result = llm.generate_text(prompt=p, temperature=0.1)
    print(llm_result)
    llm_result = await llm.agenerate_text(prompt=p, temperature=0.1)
    print(llm_result)


if __name__ == '__main__':
    asyncio.run(main())
