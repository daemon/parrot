__all__ = ['QuestionAnswering', 'StoryGeneration']

import re
from typing import List, Any, Dict

from .base import Task


class StoryGeneration(Task):
    def create_task_instruction(self) -> str:
        return (f'Create {self.num_rows} prompts to an LLM for generating stories while following the above instructions. '
                f'Only generate prompts. '
                f'Return it as a list of strings in the format ["...", "...", ...], and do not explain.')

    async def on_example(self, evaluator, llm, example: str, final_results: 'EvaluationResults',
                         running_results: List[Dict[str, Any]]):
        response = await llm.chat_complete([{'role': 'user', 'content': example}])
        grade = await evaluator.chat_complete([{
            'role': 'user',
            'content': f'Given the instruction "{example}", grade the quality of this response:'
                       f' "{response}" Respond with an integer from 1 to 5, with 5 being highest; do not explain.'}
        ])

        score = int(re.match(r'^.*([1-5]).*$', grade).group(1))
        final_results.log_row(self.name, example, response, score)
        running_results.append(dict(mos=score))

    async def on_finalize(self, evaluator, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        accuracy = sum(result['mos'] for result in running_results) / len(running_results)
        final_results.log_result(self.name, 'mos', accuracy)


class QuestionAnswering(Task):
    def create_task_instruction(self) -> str:
        return (f'Create {self.num_rows} Q&A dataset-type questions while following the above instructions. '
                f'Only generate questions, not answering. '
                f'Return it as a list of strings in the format ["...", "...", ...], and do not explain.')

    async def on_example(self, evaluator, llm, example: str, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        response = await llm.chat_complete([{'role': 'user', 'content': example}])
        grade = await evaluator.chat_complete([{
            'role': 'user',
            'content': f'Given the question "{example}", grade the factuality of this answer:'
                       f' "{response}"? Respond with an integer from 1 to 5, with 5 being most correct; do not explain.'}
        ])

        score = int(re.match(r'^.*([1-5]).*$', grade).group(1))
        final_results.log_row(self.name, example, response, score)
        running_results.append(dict(mos=score))

    async def on_finalize(self, evaluator, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        accuracy = sum(result['mos'] for result in running_results) / len(running_results)
        final_results.log_result(self.name, 'mos', accuracy)