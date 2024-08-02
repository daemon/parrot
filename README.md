# Parrot: A Fluent LLM Evaluation Framework

## Getting Started

### Quick Start

1. For a non-editable version: `pip install git+https://github.com/daemon/parrot`
2. For an editable version, clone the repository and install it with `pip install -e .`
3.

```python
import argparse
import asyncio

import parrot
import parrot.tasks


async def amain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', '-o', type=str, default='tech-results')
    parser.add_argument('--num-rows-per-task', '-n', type=int, default=20)
    args = parser.parse_args()

    gpt_4o_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt4o.json'))
    gpt_35_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt35.json'))

    with gpt_4o_llm.as_evaluator('profiles/tech.json') as evaluator:
        evaluator.add_task(parrot.tasks.QuestionAnswering(
            subcategory='retrocomputing',
            name='hard software q&a',
            num_rows=args.num_rows_per_task,
            difficulty='nearly impossible',
        ))

        evaluator.add_task(parrot.tasks.QuestionAnswering(
            subcategory='retrocomputing',
            name='easy software q&a',
            num_rows=args.num_rows_per_task,
            difficulty='very easy',
        ))

        # Add any kind of kwargs to build the task
        evaluator.add_task(parrot.tasks.SentimentClassification(
            subcategory='web services',
            creativity='very high',
            num_rows=args.num_rows_per_task,
            name='creative web services sentiment classification',
            difficulty='very challenging'
        ))

        evaluator.add_task(parrot.tasks.DocumentClassification(
            subcategory='web services',
            creativity='very high',
            literary_difficulty='very challenging',
            long_prose=True,
            num_rows=args.num_rows_per_task,
            name='creative web services document classification',
            difficulty='very challenging'
        ))

        evaluator.add_task(parrot.tasks.StoryGeneration(
            name='creative hardware text generation',
            subcategory='hardware',
            num_rows=args.num_rows_per_task,
            difficulty='extremely difficult and collegial'
        ))

        # Run the evaluation
        results = await evaluator.evaluate(gpt_35_llm)

    # evaluator.add_task(parrot.tasks.QuestionAnswering())  # Error! Immutable evaluator for safety reasons
    print(results)

    results.save(args.output_folder)


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
```
