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

## What I Learned

TBD — will be updated after all experiments are complete.

---

## Stack

Python · Ollama · sentence-transformers · pandas · matplotlib

---

## Author

[Honaxen](https://github.com/Honaxen)