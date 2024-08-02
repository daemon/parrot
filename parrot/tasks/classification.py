__all__ = ['SentimentClassification', 'DocumentClassification']

import re
from typing import List, Any, Dict

from .base import Task


class SentimentClassification(Task):
    def create_task_instruction(self) -> str:
        return (f'Create {self.num_rows} sentiment dataset-type sentences while following the above instructions. '
                f'Only generate sentences which have positive, neutral, or negative polarity. '
                f'Return it as a list of strings in the format ["...", "...", ...], and do not explain.')

    async def on_example(self, evaluator, llm, example: str, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        response = await llm.chat_complete([{
            'role': 'user',
            'content': f'Grade the sentiment of the following sentence as one of negative, neutral, or positive: "{example}". Do not explain.'
        }])

        grade = await evaluator.chat_complete([{
            'role': 'user',
            'content': f'Given the sentence "{example}", is the sentiment polarity "{response}"? Respond with only "yes" or "no"; do not explain.'}
        ])

        try:
            response = re.match(r'^.*(neutral|positive|negative).*$', response.lower()).group(1)
        except:  # hacky catch-all to save time...
            pass

        score = float('yes' in grade.lower())
        final_results.log_row(self.name, example, response, score)
        running_results.append(dict(accuracy=score))

    async def on_finalize(self, evaluator, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        accuracy = sum(result['accuracy'] for result in running_results) / len(running_results)
        final_results.log_result(self.name, 'accuracy', accuracy)


class DocumentClassification(Task):
    def create_task_instruction(self) -> str:
        return (f'Create {self.num_rows} passages while following the above instructions. '
                f'Each passage should have 3-5 sentences and be about a different topic. '
                f'Return it as a list of JSON strings in the format ["...", "...", ...], and do not explain.')

    async def on_example(self, evaluator, llm, example: str, final_results: 'EvaluationResults',
                         running_results: List[Dict[str, Any]]):
        response = await llm.chat_complete([{
            'role': 'user',
            'content': f'What is the topic of the following passage: "{example}"? Do not explain.'
        }])

        grade = await evaluator.chat_complete([{
            'role': 'user',
            'content': f'Given the passage "{example}", is the topic of the passage "{response}"? Respond with only "yes" or "no"; do not explain.'}
        ])

        score = float('yes' in grade.lower())
        final_results.log_row(self.name, example, response, score)
        running_results.append(dict(accuracy=score))

    async def on_finalize(self, evaluator, final_results: 'EvaluationResults', running_results: List[Dict[str, Any]]):
        accuracy = sum(result['accuracy'] for result in running_results) / len(running_results)
        final_results.log_result(self.name, 'accuracy', accuracy)