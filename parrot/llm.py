__all__ = ['LLM', 'AzureOpenAILLM', 'AzureOpenAICredentials', 'HFInferenceEndpointsLLM', 'HuggingFaceCredentials']

import os
from typing import Dict, List

import openai
import pydantic
import requests


class AzureOpenAICredentials(pydantic.BaseModel):
    azure_endpoint: str
    api_key: str
    azure_deployment: str
    api_version: str


class HuggingFaceCredentials(pydantic.BaseModel):
    chat_formatter: str
    inference_endpoint: str
    api_key: str


class LLM:
    async def chat_complete(self, conversation: List[Dict[str, str]], timeout_secs: int = 60) -> str:
        raise NotImplementedError

    async def prompt_complete(self, prompt: str, timeout_secs: int = 60) -> str:
        raise NotImplementedError

    def can_chat(self) -> bool:
        return False

    def can_prompt(self) -> bool:
        return False


class AzureOpenAILLM(LLM):
    def __init__(self, credentials: AzureOpenAICredentials):
        self.client = openai.AsyncAzureOpenAI(**credentials.model_dump())
        self.credentials = credentials

    async def chat_complete(self, conversation: List[Dict[str, str]], timeout_secs: int = 60) -> str:
        return (await self.client.chat.completions.create(
            model=self.credentials.azure_deployment,
            messages=conversation,
            timeout=timeout_secs,
            temperature=0.0,
        )).choices[0].message.content

    async def prompt_complete(self, prompt: str, timeout_secs: int = 60) -> str:
        return await self.chat_complete([{'role': 'system', 'content': prompt}], timeout_secs)

    def can_chat(self) -> bool:
        return True

    def can_prompt(self) -> bool:
        return True


class SimpleLlamaChatFormatter:
    def format(self, conversation: List[Dict[str, str]]) -> str:
        strings = []

        for message in conversation:
            match message['role']:
                case 'system':
                    strings.append(f'''<s> [INST] <<SYS>> {message['content']} <</SYS>>''')
                case 'user':
                    if not strings:
                        strings.append(f'''<s> [INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

                                       If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>''')

                    strings.append(f'''{message['content']} [/INST]''')

        return ' '.join(strings)

    def extract(self, output: str) -> str:
        return output.split('[/INST]')[-1].strip()


class HFInferenceEndpointsLLM(LLM):
    def __init__(self, credentials: HuggingFaceCredentials):
        self.credentials = credentials

        match credentials.chat_formatter:
            case 'llama':
                self.formatter = SimpleLlamaChatFormatter()
            case _:
                raise ValueError('Unknown model_tokenizer')

    async def chat_complete(self, conversation: List[Dict[str, str]], timeout_secs: int = 60) -> str:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.credentials.api_key}',
            'Content-Type': 'application/json'
        }

        async def query(payload):
            response = requests.post(self.credentials.inference_endpoint, headers=headers, json=payload)
            return response.json()[0]['generated_text']

        output = await query(dict(inputs=self.formatter.format(conversation), parameters=dict(max_new_tokens=1000)))

        return self.formatter.extract(output)
