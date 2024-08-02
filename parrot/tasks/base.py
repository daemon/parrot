__all__ = ['Task']

import asyncio
from typing import Dict, Any, List

from tqdm.asyncio import tqdm_asyncio


KWARGS_MEANING_MAP = {
    'subcategory': 'The topic is {value}.',
    'difficulty': 'The task difficulty is {value}.',
}


class Task:
    """
    An abstract base class that represents a task to be executed by the evaluator for evaluating LLMs with LLMs. It
    encapsulates the task lifecycle and logic, such as prompt initialization, query generation, and finalization.
    Subclasses _must_ implement :py:meth:`.create_task_instruction` and :py:meth:`.on_query`.

    See Also:
        - :py:class:`.SentimentClassification`
        - :py:class:`.DocumentClassification`
    """
    default_name = 'task'

    def __init__(
            self,
            name: str = None,
            num_rows: int = 100,
            num_parallel: int = 4,
            **kwargs
    ):
        """
        Instantiates a new task with the give name, number of rows to generate, the number of parallel `asyncio`
        coroutines to run, and keyword arguments to parameterize the task. The keyword arguments can be anything to feed
        into the evaluator LLM for parameterizing the task, with special meanings for some keys.

        Specifically, a task is a set of callbacks, logic, and instructions wrapped around a set of generated queries
        that are used to evaluate an LLM. The task is responsible for generating the queries, providing instructions to
        the evaluator, and storing the results to an :py:class:`.EvaluatorResults` object.

        Args:
            name (`str`, optional): The name of the task. Defaults to `None`.
            num_rows (`int`, optional): The number of rows to generate. Defaults to `100`.
            num_parallel (`int`, optional): The number of parallel `asyncio` coroutines to run. Defaults to `4`.
            **kwargs: Keyword arguments to parameterize the task. `subcategory` and `difficulty` have special meanings
                of expanding into the forms `The topic is {value}.` and `The task difficulty is {value}.`, respectively.
        """
        self.name = name or self.default_name
        self.kwargs = kwargs
        self.num_rows = num_rows
        self.prefix_message = ''
        self.examples = []
        self.semaphore = asyncio.Semaphore(num_parallel)

    async def initialize(self, evaluator):
        """
        Initializes the task by generating the queries and setting the prefix message for the evaluator. For the sake
        of simplicity (this is incredibly ill-advised), the examples are assumed to be generated as a Python
        `eval`-able string.

        Args:
            evaluator (`parrot.Evaluator`): The evaluator to initialize the task for.
        """
        messages = []

        for key, value in self.kwargs.items():
            if key in KWARGS_MEANING_MAP:
                messages.append(KWARGS_MEANING_MAP[key].format(value=value))
            else:
                messages.append(f'The task has a parameter "{key}" with value "{value}".')

        self.prefix_message = ' '.join(messages)
        instruction = self.create_task_instruction()
        self.examples = eval(await self.generate(evaluator, {'role': 'user', 'content': instruction}))

    def create_task_instruction(self) -> str:
        """The task instruction for generating input examples."""
        raise NotImplementedError

    async def generate(self, evaluator, message: Dict[str, str]) -> str:
        """

        Args:
            evaluator (`parrot.Evaluator`): The evaluator to use for chat completion (generation).
            message (`Dict[str, str]`): The OpenAI-style input.

        Returns:
            str: The generated response from the evaluator.
        """
        messages = [dict(role='user', content=self.prefix_message)]

        if messages[0]['role'] == 'user':
            messages[0]['content'] += f' {message["content"]}'
        else:
            messages.append(message)

        return await evaluator.chat_complete(messages)

    async def on_example(self, evaluator, llm, example: str, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        """
        The callback to implement for running the task on a single example. This is called for each example in the
        examples instance variable. This method should do the following things:

        1. Generate a response from the target LLM `llm`.
        2. Parse the response if necessary and use the evaluator LLM `evaluator` to judge the response.
        3. Store the results in the final_results and running_results objects, where `final_results` is the final evaluator
        results and `running_results` is a list of intermediate temporary results tracked throughout the task run loop.

        Args:
            evaluator (`parrot.Evaluator`): The evaluator to use for chat completion (generation).
            llm (`parrot.LLM`): The LLM to evaluate.
            example (`str`): The example to evaluate.
            final_results (`parrot.EvaluationResults`): The final results object to store the results in.
            running_results (`List[Dict[str, Any]]`): The running results object to store intermediate results in.
        """
        raise NotImplementedError

    async def on_finalize(self, evaluator, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        """
        The callback to implement for finalizing the task. This is called with the final results and the intermediate
        running results after all the examples have been run. If implemented, this method should aggregate any
        remaining intermediate results in `running_results` and store them in `final_results`.
        """
        pass

    async def run(self, evaluator, llm, semaphore: asyncio.Semaphore = None) -> 'EvaluationResults':
        """
        Runs the task with the given evaluator and LLM. This method initializes the task, runs the examples, and finalizes
        the task. The final results are returned.

        Args:
            evaluator (`parrot.Evaluator`): The evaluator to use for chat completion (generation).
            llm (`parrot.LLM`): The LLM to evaluate.
            semaphore (`asyncio.Semaphore`, optional): The semaphore to use instead of the task's default semaphore. This
                is used if a semaphore should be shared among multiple tasks. Defaults to `None`.
        """
        from ..llm import EvaluationResults

        async def evaluate(semaphore: asyncio.Semaphore, query: str, final_results: EvaluationResults, running_results: List[Dict[str, Any]]):
            async with semaphore:
                await self.on_example(evaluator, llm, query, final_results, running_results)

        final_results = EvaluationResults()
        running_results = []
        semaphore = semaphore or self.semaphore
        coroutines = []

        for query in self.examples:
            coroutines.append(evaluate(semaphore, query, final_results, running_results))

        await tqdm_asyncio.gather(*coroutines, desc=f'Running {self.name}')
        await self.on_finalize(evaluator, final_results, running_results)

        return final_results
