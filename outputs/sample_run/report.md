# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpotqa_diverse_100.json
- Mode: mock
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.76 | 0.94 | 0.18 |
| Avg attempts | 1 | 1.39 | 0.39 |
| Avg token estimate | 385 | 733.8 | 348.8 |
| Avg latency (ms) | 48934.62 | 58534.21 | 9599.59 |

## Failure modes
```json
{
  "react": {
    "none": 76,
    "wrong_final_answer": 24
  },
  "reflexion": {
    "none": 94,
    "wrong_final_answer": 6
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
