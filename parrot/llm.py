__all__ = ['LLM', 'AzureOpenAILLM', 'AzureOpenAICredentials', 'HFInferenceEndpointsLLM', 'HuggingFaceCredentials',
           'EvaluationResults', 'Evaluator']

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import openai
import pandas as pd
import pydantic
import requests


class EvaluationResults:
    def __init__(self):
        self.rows = []
        self.results = defaultdict(dict)

    def log_row(self, task_name: str, input: str, output: str, score: Any):
        row = {}
        row['task'] = task_name
        row['input'] = input
        row['output'] = output
        row['score'] = score

        self.rows.append(row)

    def log_result(self, task_name: str, key: str, value: Any):
        self.results[task_name][key] = value

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)
        df.to_csv(path / 'dataset.tsv', index=False, sep='\t')

        with open(path / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=4)

    def __str__(self) -> str:
        return repr(self.results)

    def __repr__(self) -> str:
        return repr(self.results)

    def __iadd__(self, other: 'EvaluationResults') -> 'EvaluationResults':
        self.rows += other.rows
        self.results.update(other.results)

        return self

    def __add__(self, other: 'EvaluationResults') -> 'EvaluationResults':
        new_results = EvaluationResults()
        new_results += self
        new_results += other

        return new_results


class AzureOpenAICredentials(pydantic.BaseModel):
    """Credentials data model for the Azure OpenAI API."""
    azure_endpoint: str
    api_key: str
    azure_deployment: str
    api_version: str


class HuggingFaceCredentials(pydantic.BaseModel):
    """Credentials data model for the HuggingFace Inference Endpoints API."""
    chat_formatter: str
    inference_endpoint: str
    api_key: str


class LLM:
    """
    Abstract base class representing a large language model capable of either chat or prompt completion, or both.
    Subclasses should implement `chat_complete` or `prompt_complete` as appropriate, and override `can_chat` and
    `can_prompt`. Not the most elegant solution, but it works for now.
    """
    async def chat_complete(self, conversation: List[Dict[str, str]], timeout_secs: int = 60) -> str:
        """
        Complete a chat conversation using the LLM, given a conversation represented as a list of OpenAI-style
        messages and a timeout in seconds.

        Args:
            conversation: A list of messages in the format {'role': str, 'content': str}.
            timeout_secs: The maximum time to wait for completion, in seconds, before an exception is thrown.

        Returns:
            The completion of the conversation as a string.

        Raises:
            NotImplementedError: If the LLM does not support chat completion.
            Exception: Some other exception that occurred during completion, including timeout. This is a catch-all.
        """
        raise NotImplementedError

    async def prompt_complete(self, prompt: str, timeout_secs: int = 60) -> str:
        """Prompt completion counterpart to `chat_complete`."""
        raise NotImplementedError

    def can_chat(self) -> bool:
        """Whether the LLM can be used for chat completion."""
        return False

    def can_prompt(self) -> bool:
        """Whether the LLM can be used for prompt completion."""
        return False

    def as_evaluator(self, profile: str | Path = None) -> 'Evaluator':
        """Create an evaluator using this LLM and the given profile."""
        evaluator = Evaluator(self)
        profile = json.loads(Path(profile).read_text()) if profile else {}
        evaluator.system_message = profile.get('system_message', '')
        evaluator.prefix_message = profile.get('prefix_message', '')

        return evaluator


class AzureOpenAILLM(LLM):
    """Large language model implementation using the Azure OpenAI API."""
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


class LlamaChatFormatter:
    """Formats and extracts chat conversations for LLaMA."""
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


class MistralChatFormatter:
    """Formats and extracts chat conversations for Mistral."""
    def format(self, conversation: List[Dict[str, str]]) -> str:
        strings = []

        for message in conversation:
            match message['role']:
                case 'system':
                    strings.append(f'''<s> [INST] {message['content']} ''')
                case 'user':
                    if not strings:
                        strings.append(f'''<s> [INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

                                       If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.''')

                    strings.append(f'''{message['content']} [/INST]''')

        return ' '.join(strings)

    def extract(self, output: str) -> str:
        return output.split('[/INST]')[-1].strip()


class HFInferenceEndpointsLLM(LLM):
    """
    Large language model implementation using the HuggingFace Inference Endpoints API. Only supports chat completion
    for the LLaMA and Mistral family of models.
    """
    def __init__(self, credentials: HuggingFaceCredentials):
        self.credentials = credentials

        match credentials.chat_formatter:
            case 'llama':
                self.formatter = LlamaChatFormatter()
            case 'mistral':
                self.formatter = MistralChatFormatter()
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

    def can_chat(self) -> bool:
        return True



class Evaluator(LLM):
    """
    A class for evaluating the performance of an LLM on a set of tasks driven by another LLM as a judge.
    Once the evaluator context is entered, the system and prefix messages are made immutable, thus preventing
    unintentional changes at evaluation time. Evaluators are also one-time use: once an evaluator has
    finished evaluating, its context cannot be reentered.
    """
    def __init__(self, llm: LLM):
        self.llm = llm
        self.mutable = True
        self._system_message = ''
        self._prefix_message = ''
        self.tasks: List['Task'] = []

    def add_task(self, task: 'Task') -> 'Evaluator':
        self.tasks.append(task)
        return self

    @property
    def system_message(self) -> str:
        return self._system_message

    @system_message.setter
    def system_message(self, system_message: str):
        assert self.mutable, 'Cannot set system_message after evaluation has started'
        self._system_message = system_message

    @property
    def prefix_message(self) -> str:
        return self._prefix_message

    @prefix_message.setter
    def prefix_message(self, prefix_message: str):
        assert self.mutable, 'Cannot set prefix_message after evaluation has started'
        self._prefix_message = prefix_message

    async def evaluate(self, llm: LLM) -> 'EvaluationResults':
        results = EvaluationResults()

        for task in self.tasks:
            await task.initialize(self)
            results += await task.run(self, llm)

        return results

    async def chat_complete(self, conversation: List[Dict[str, str]], timeout_secs=60):
        conversation = conversation.copy()
        prefix = []

        if self.system_message:
            prefix.append(dict(role='system', content=self.system_message))

        if self.prefix_message:
            prefix.append(dict(role='user', content=self.prefix_message))

        conversation = prefix + conversation

        return await self.llm.chat_complete(conversation, timeout_secs)

    async def prompt_complete(self, prompt, timeout_secs: int = 60):
        return await self.llm.prompt_complete(prompt, timeout_secs)

    def can_chat(self):
        return self.llm.can_chat()

    def can_prompt(self):
        return self.llm.can_prompt()

    def __enter__(self):
        assert self.mutable, 'Cannot re-enter evaluation context'
        self.mutable = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
