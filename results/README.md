# Proxy Results for IMDB and GSM8K

As evaluated by GPT-4o:

| Model       | IMDB | GSM8K | Truth IMDB | Truth GSM8K |
|-------------|------| ---- |-----------|-------------|
| GPT-4o-mini | 100  | 4.6  | -         | -           |
| Mistral 0.3 7B | 90 | 3.6 | 96        | 38.1        |
| GPT-3.5 Turbo (0613) | 90 | 3.3 | 94.3      | 46.9        |
| LLaMA 2 7B | 80 | 1.4 | 90.7      | 13.3        |

Spearman's rho between the model's proxy quality and the true quality is 0.68 ($p < 0.05$), indicating a strong correlation.