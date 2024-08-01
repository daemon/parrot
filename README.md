# Parrot: The Fluent LLM Evaluation Framework

## Getting Started

### Quick Start

1. For a non-editable version: `pip install git+https://github.com/daemon/parrot`
2. For an editable version, clone the repository and install it with `pip install -e .`
3. 
```python
import asyncio

import parrot
import parrot.tasks


async def amain():
    tech_parrot = parrot.Parrot(profile='profiles/tech.json', llm=parrot.AzureOpenAILLM(base_model='gpt-4o'))
    llm = parrot.AzureOpenAILLM(base_model='gpt-3.5')
    
    with tech_parrot.evaluator() as evaluator:
        evaluator.add_task(parrot.tasks.QuestionAnswering(name='tech q&a'), n=100)  # 100 rows; n is optional
        
        # Add any kind of kwargs to build the task
        evaluator.add_task(parrot.tasks.SentimentClassification(subcategory='software')) 
        evaluator.add_task(parrot.tasks.TextGeneration(subcategory='hardware', difficulty='hard'))
        
        # Run the evaluation
        results = await evaluator.evaluate(llm)
    
    # evaluator.add(parrot.tasks.QuestionAnswering())  # Error! Immutable evaluator
    print(results)
    
    results.save('tech-results/')

    
def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
```
