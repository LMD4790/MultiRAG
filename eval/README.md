# multimodel_RAG Evaluation

This folder contains an offline evaluation runner for the tourism RAG agents.

## Case Format

Each JSONL row may contain:

```json
{
  "id": "case_001",
  "question": "香港太平山顶有哪些看点？",
  "expected_agent": "multi_model_agent",
  "expected_keywords": ["太平山", "香港"],
  "forbidden_keywords": ["兰州"],
  "reference_answer": "应回答香港太平山顶、维港夜景等相关内容。"
}
```

## Modes

- `router`: only predict route with DeepSeek.
- `expected_agent`: call the agent named by `expected_agent`.
- `system`: call the full `System_Agent.compiled_workflow`.

## Commands

Route-only evaluation:

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode router
```

Call expected specialist agents and use DeepSeek judge:

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode expected_agent --judge
```

Call the full system workflow:

```bash
python eval/run_eval.py --cases eval/cases/sample_cases.jsonl --mode system --judge
```

Outputs:

- `eval/outputs/eval_results.jsonl`
- `eval/reports/eval_summary.json`
