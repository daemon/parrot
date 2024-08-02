import argparse
import asyncio

import parrot
import parrot.tasks


async def amain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', '-o', type=str, default='helm-results')
    parser.add_argument('--num-rows-per-task', '-n', type=int, default=20)
    parser.add_argument('--profile', '-p', type=str, default='profiles/general.json')
    parser.add_argument('--target-llm', '-t', type=str, default='gpt35', choices=['gpt4o', 'gpt4o-mini', 'gpt35', 'llama-2-7b', 'mistral-7b'])
    args = parser.parse_args()

    gpt_4o_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt4o.json'))
    gpt_4o_mini_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt4o-mini.json'))
    gpt_35_llm = parrot.AzureOpenAILLM(parrot.AzureOpenAICredentials.parse_file('gpt35.json'))
    llama_2_7b = parrot.HFInferenceEndpointsLLM(parrot.HuggingFaceCredentials.parse_file('hf-llama-2-7b.json'))
    mistral_7b = parrot.HFInferenceEndpointsLLM(parrot.HuggingFaceCredentials.parse_file('hf-mistral-7b.json'))

    match args.target_llm:
        case 'gpt4o':
            evaluate_llm = gpt_4o_llm
        case 'gpt4o-mini':
            evaluate_llm = gpt_4o_mini_llm
        case 'gpt35':
            evaluate_llm = gpt_35_llm
        case 'llama-2-7b':
            evaluate_llm = llama_2_7b
        case 'mistral-7b':
            evaluate_llm = mistral_7b
        case _:
            raise ValueError(f'Unknown target LLM: {args.target_llm}')

    with gpt_4o_llm.as_evaluator(profile=args.profile) as evaluator:
        evaluator.add_task(parrot.tasks.QuestionAnswering(
            subcategory='grade school math questions',
            name='gsm8k hard',
            num_rows=args.num_rows_per_task,
            difficulty='very difficult',
            numerical_answer_and_reasoning=True,
        )).add_task(parrot.tasks.SentimentClassification(
            subcategory='movie reviews',
            name='imdb-like movie reviews',
            num_rows=args.num_rows_per_task,
            difficulty='very hard even for humans',
        ))

        # Run the evaluation
        results = await evaluator.evaluate(evaluate_llm)

    # evaluator.add_task(parrot.tasks.QuestionAnswering())  # Error! Immutable evaluator
    print(results)

    results.save(args.output_folder)


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()