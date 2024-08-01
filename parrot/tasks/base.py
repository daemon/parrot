__all__ = ['Task']

import asyncio
from typing import Dict, Any, List

from tqdm.asyncio import tqdm_asyncio

KWARGS_MEANING_MAP = {
    'subcategory': 'The topic is {value}.',
    'difficulty': 'The task difficulty is {value}.',
}


class Task:
    default_name = 'task'

    def __init__(
            self,
            name: str = None,
            num_rows: int = 100,
            num_parallel: int = 4,
            **kwargs
    ):
        self.name = name or self.default_name
        self.kwargs = kwargs
        self.num_rows = num_rows
        self.prefix_message = ''
        self.queries = []
        self.semaphore = asyncio.Semaphore(num_parallel)

    async def initialize(self, evaluator):
        messages = []

        for key, value in self.kwargs.items():
            if key in KWARGS_MEANING_MAP:
                messages.append(KWARGS_MEANING_MAP[key].format(value=value))
            else:
                messages.append(f'The task has a parameter "{key}" with value "{value}".')

        self.prefix_message = ' '.join(messages)
        instruction = self.create_task_instruction()
        self.queries = eval(await self.generate(evaluator, {'role': 'user', 'content': instruction}))

    def create_task_instruction(self) -> str:
        raise NotImplementedError

    async def generate(self, evaluator, message: Dict[str, str]):
        messages = [dict(role='user', content=self.prefix_message)]

        if messages[0]['role'] == 'user':
            messages[0]['content'] += f' {message["content"]}'
        else:
            messages.append(message)

        return await evaluator.chat_complete(messages)

    async def on_query(self, evaluator, llm, query: str, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        raise NotImplementedError

    async def on_finalize(self, evaluator, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        pass

    async def run(self, evaluator, llm, semaphore: asyncio.Semaphore = None) -> 'EvaluationResults':
        from ..parrot import EvaluationResults

        async def evaluate(semaphore: asyncio.Semaphore, query: str, final_results: EvaluationResults, running_results: List[Dict[str, Any]]):
            async with semaphore:
                await self.on_query(evaluator, llm, query, final_results, running_results)

        final_results = EvaluationResults()
        running_results = []
        semaphore = semaphore or self.semaphore
        coroutines = []

        for query in self.queries:
            coroutines.append(evaluate(semaphore, query, final_results, running_results))

        await tqdm_asyncio.gather(*coroutines, desc=f'Running {self.name}')
        await self.on_finalize(evaluator, final_results, running_results)

        return final_results
