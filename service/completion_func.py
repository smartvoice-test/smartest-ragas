import os
import uuid
import json
import asyncio
import typing as t

import httpx
from dotenv import load_dotenv, find_dotenv
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ragas.llms.base import BaseRagasLLM
from ragas.llms.prompt import PromptValue

_ = load_dotenv(find_dotenv())
_authorization = os.getenv("OpenAPIKey")


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

    def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks: Callbacks = [],
    ) -> LLMResult:
        ...

    async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks: Callbacks = [],
    ) -> LLMResult:
        completions = []
        messages = []
        for row in prompt.to_messages():
            if isinstance(row, SystemMessage):
                messages.append({"role": "system", "content": SystemMessage.content})
            if isinstance(row, HumanMessage):
                messages.append({"role": "user", "content": HumanMessage.content})
            if isinstance(row, AIMessage):
                messages.append({"role": "assistant", "content": AIMessage.content})
        async with httpx.AsyncClient(verify=False) as client:
            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "ApiName": "robotGptLLMApi",
                    "Model": self.model,
                    "Authorization": _authorization
                }
                async with client.stream(
                        "POST",
                        self.address,
                        headers=headers,
                        json={
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
                            "messages": prompt.to_messages(),
                            "trace_id": f"RAGAS{uuid.uuid4().hex}@cloudminds-test.com.cn"
                        },
                        timeout=self.api_timeout,
                ) as response:
                    async for chunk in response.aiter_text():
                        chunk = chunk.strip() if isinstance(chunk, str) else chunk
                        if not chunk:
                            continue
                        data = json.loads(chunk.removeprefix("data:"))
                        print(data)
            except Exception as e:
                print(e)


async def main():
    # https://dataai.harix.iamidata.com/llm/api/ask
    llm = GatewayLLM(model="robotgpt-qw-1dot8")
    p = PromptValue(prompt_str="你好")
    llm_result = await llm.agenerate_text(prompt=p)
    print(llm_result)


if __name__ == '__main__':
    asyncio.run(main())
