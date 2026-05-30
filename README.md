# LLM Evaluation Playground

A systematic framework for analyzing LLM behavior.
Not just using models — understanding how they fail.

---

## Why This Matters

Anyone can call an API and get a response.
This project goes deeper:
- How consistent is the model across different prompts?
- Where does it hallucinate?
- How does prompt design affect output quality?
- How do different models compare on the same task?

---

## Evaluation Dimensions

| Dimension | What We Measure |
|-----------|----------------|
| Hallucination | Does the model make up facts? |
| Prompt Sensitivity | How much does wording affect the answer? |
| Response Scoring | Is the answer relevant and accurate? |
| Latency | How fast does the model respond? |
| Retrieval Quality | Does RAG grounding improve accuracy? |

---

## Project Structure

```
llm-evaluation-playground/
├── notebooks/
│   ├── 01_hallucination_eval.ipynb
│   ├── 02_prompt_sensitivity.ipynb
│   ├── 03_response_scoring.ipynb
│   └── 04_model_comparison.ipynb
├── src/
│   └── evaluator.py
├── results/
└── README.md
```

---

## Model

All experiments use local models via Ollama — no API key required.

---

## Results

| Experiment | Key Finding |
|------------|-------------|
| Hallucination | 0% hallucination with clear context instructions |
| Prompt Sensitivity | 90% consistency across 4 question variants |
| Response Scoring | Average score 0.60 — short answers penalized |
| Prompt Strategy | Chain-of-thought (0.89) > Zero-shot (0.71) > One-shot (0.63) |

---

## What I Learned

Evaluating LLMs is not about running one test — it is about systematic measurement.

Hallucination depends heavily on prompt design.
With a clear "answer only from context" instruction, gemma3:12b refused to hallucinate.

Chain-of-thought prompting consistently outperforms other strategies.
When the model reasons step by step, it produces more complete and accurate answers.

Semantic similarity scoring is useful but imperfect.
Short correct answers score lower than verbose ones — the metric needs calibration.

One-shot prompting can hurt if the example is not carefully chosen.

---

## Stack

Python · Ollama · sentence-transformers · pandas · matplotlib

---

## Author

[Honaxen](https://github.com/Honaxen)