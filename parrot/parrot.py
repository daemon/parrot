__all__ = ['Parrot', 'Evaluator', 'EvaluationResults']

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from .llm import LLM


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


class Evaluator(LLM):
    def __init__(self, llm: LLM):
        self.llm = llm
        self.mutable = True
        self._system_message = ''
        self._prefix_message = ''
        self.tasks = []

    def add_task(self, task: 'Task'):
        self.tasks.append(task)

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


class Parrot:
    def __init__(self, llm: LLM, profile: str | Path = None):
        self.llm = llm
        self.profile = json.loads(Path(profile).read_text()) if profile else {}

    def evaluator(self) -> Evaluator:
        evaluator = Evaluator(self.llm)
        evaluator.system_message = self.profile.get('system_message', '')
        evaluator.prefix_message = self.profile.get('prefix_message', '')

        return evaluator
