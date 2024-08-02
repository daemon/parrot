for MODEL in mistral-7b llama-2-7b gpt35 gpt4o-mini; do
	python -u -m parrot.run.helm_evaluate -n 10 -o helm-results-$MODEL -t $MODEL;
done
