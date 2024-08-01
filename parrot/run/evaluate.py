import argparse
import asyncio

import parrot
import parrot.tasks


async def amain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-rows-per-task', '-n', type=int, default=20)
    args = parser.parse_args()

    gpt_4o_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt4o.json'))
    gpt_4o_mini_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt4o-mini.json'))
    gpt_35_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt35.json'))
    llama_2_7b = parrot.HFInferenceEndpointsLLM(parrot.HuggingFaceCredentials.parse_file('hf-llama-2-7b.json'))

    tech_parrot = parrot.Parrot(profile='profiles/tech.json', llm=gpt_4o_llm)

    with tech_parrot.evaluator() as evaluator:
        # 100 rows; n is optional
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

        # evaluator.add_task(parrot.tasks.TextGeneration(subcategory='hardware', difficulty='hard'))

        # Run the evaluation
        results = await evaluator.evaluate(llama_2_7b)

    # evaluator.add(parrot.tasks.QuestionAnswering())  # Error! Immutable evaluator
    print(results)

    results.save('tech-results/')


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()