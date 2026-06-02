from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_env, load_project_env, require_env
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage


load_project_env()


@dataclass
class EvalResult:
    mode: str
    case_id: str
    question: str
    expected_agent: str | None
    called_agent: str | None
    predicted_agent: str | None
    answer: str
    keyword_hit_rate: float
    forbidden_hit: bool
    judge: dict[str, Any]
    latency_seconds: float
    error: str | None = None


ROUTER_PROMPT = """You are evaluating a tourism RAG router.
Choose exactly one route for the user's question:
- rag_agent: mainland China basic tourism questions, itinerary, attractions, transport, food, lodging.
- graphrag_agent: mainland China higher-level comparison, summary, theme analysis, multi-attraction relationship analysis.
- multi_model_agent: Hong Kong, Macau, or Taiwan tourism questions.

Return only one of: rag_agent, graphrag_agent, multi_model_agent.

Question: {question}
"""


JUDGE_PROMPT = """You are a strict evaluator for a tourism RAG answer.
Score the answer against the question and reference hints.

Question:
{question}

Expected agent:
{expected_agent}

Expected keywords:
{expected_keywords}

Forbidden keywords:
{forbidden_keywords}

Reference answer, if any:
{reference_answer}

Actual answer:
{answer}

Return strict JSON only with this schema:
{{
  "correctness": 1-5,
  "faithfulness": 1-5,
  "completeness": 1-5,
  "relevance": 1-5,
  "clarity": 1-5,
  "hallucination": true/false,
  "pass": true/false,
  "reason": "short Chinese explanation"
}}
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def get_chat_model(model_env: str = "JUDGE_MODEL"):
    provider = get_env("JUDGE_PROVIDER", "deepseek").lower()
    if provider == "openai":
        model_name = get_env(model_env, get_env("OPENAI_JUDGE_MODEL", "gpt-5.5"))
        return init_chat_model(
            model=model_name,
            model_provider="openai",
            api_key=require_env("OPENAI_API_KEY"),
            base_url=get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=0,
        )

    model_name = get_env(model_env, get_env("DEEPSEEK_JUDGE_MODEL", get_env("DEEPSEEK_MODEL", "deepseek-chat")))
    return init_chat_model(
        model=model_name,
        model_provider="deepseek",
        api_key=require_env("DEEPSEEK_API_KEY"),
        base_url=get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )


async def predict_route(question: str) -> str:
    model = get_chat_model("ROUTER_MODEL")
    resp = await model.ainvoke([{"role": "user", "content": ROUTER_PROMPT.format(question=question)}])
    text = resp.content.strip()
    for agent in ("multi_model_agent", "graphrag_agent", "rag_agent"):
        if agent in text:
            return agent
    return text


async def call_agent(agent_name: str, question: str) -> str:
    state = {
        "messages": [HumanMessage(content=question)],
        "generate_times": 0,
    }
    if agent_name == "rag_agent":
        from Naive_RAG.rag_agent import rag_agent

        result = await rag_agent.ainvoke(state)
    elif agent_name == "graphrag_agent":
        from GraphRAG.graphrag_agent import graphrag_agent

        result = await graphrag_agent.ainvoke(state)
    elif agent_name == "multi_model_agent":
        from vlm.multi_model_agent import multi_model_agent

        result = await multi_model_agent.ainvoke(state)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
        if getattr(msg, "type", None) == "ai":
            return msg.content
    return str(result)


async def call_system(question: str) -> tuple[str | None, str]:
    from System_Agent import compiled_workflow

    result = await compiled_workflow.ainvoke(
        {
            "messages": [HumanMessage(content=question)],
            "next_node": "",
            "generate_times": 0,
        }
    )
    predicted = result.get("next_node")
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            return predicted, msg.content
        if getattr(msg, "type", None) == "ai":
            return predicted, msg.content
    return predicted, str(result)


def keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    hits = sum(1 for kw in expected_keywords if kw and kw in answer)
    return hits / len(expected_keywords)


def has_forbidden(answer: str, forbidden_keywords: list[str]) -> bool:
    return any(kw for kw in forbidden_keywords if kw and kw in answer)


async def judge_case(case: dict[str, Any], answer: str) -> dict[str, Any]:
    model = get_chat_model("JUDGE_MODEL")
    prompt = JUDGE_PROMPT.format(
        question=case["question"],
        expected_agent=case.get("expected_agent", ""),
        expected_keywords=case.get("expected_keywords", []),
        forbidden_keywords=case.get("forbidden_keywords", []),
        reference_answer=case.get("reference_answer", ""),
        answer=answer,
    )
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    try:
        data = extract_json_object(resp.content)
    except Exception:
        data = {
            "correctness": 0,
            "faithfulness": 0,
            "completeness": 0,
            "relevance": 0,
            "clarity": 0,
            "hallucination": True,
            "pass": False,
            "reason": f"Judge returned non-JSON: {resp.content[:500]}",
        }
    return data


async def eval_one(case: dict[str, Any], mode: str, judge: bool) -> EvalResult:
    case_id = str(case.get("id", ""))
    question = case["question"]
    expected_agent = case.get("expected_agent")
    called_agent: str | None = None
    predicted_agent: str | None = None
    answer = ""
    error = None
    start = time.perf_counter()
    try:
        if mode == "router":
            predicted_agent = await predict_route(question)
            answer = predicted_agent
        elif mode == "system":
            predicted_agent, answer = await call_system(question)
        elif mode == "expected_agent":
            if not expected_agent:
                raise ValueError("expected_agent mode requires expected_agent in every case")
            called_agent = expected_agent
            answer = await call_agent(expected_agent, question)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as exc:
        error = repr(exc)
    latency = time.perf_counter() - start

    expected_keywords = case.get("expected_keywords", [])
    forbidden_keywords = case.get("forbidden_keywords", [])
    hit_rate = keyword_hit_rate(answer, expected_keywords)
    forbidden = has_forbidden(answer, forbidden_keywords)
    if judge and not error and mode != "router":
        try:
            judge_result = await judge_case(case, answer)
        except Exception as exc:
            judge_result = {
                "correctness": 0,
                "faithfulness": 0,
                "completeness": 0,
                "relevance": 0,
                "clarity": 0,
                "hallucination": True,
                "pass": False,
                "reason": "Judge call failed",
                "judge_error": repr(exc),
            }
    else:
        judge_result = {}
    return EvalResult(
        mode=mode,
        case_id=case_id,
        question=question,
        expected_agent=expected_agent,
        called_agent=called_agent,
        predicted_agent=predicted_agent,
        answer=answer,
        keyword_hit_rate=hit_rate,
        forbidden_hit=forbidden,
        judge=judge_result,
        latency_seconds=latency,
        error=error,
    )


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    ok = [r for r in results if not r.error]
    route_cases = [r for r in results if r.expected_agent and r.predicted_agent]
    route_correct = [
        r for r in route_cases if r.predicted_agent and r.expected_agent in r.predicted_agent
    ]
    judged = [r for r in results if r.judge]
    judged_pass = [r for r in judged if bool(r.judge.get("pass"))]
    hallucinations = [r for r in judged if bool(r.judge.get("hallucination"))]
    by_agent: dict[str, dict[str, Any]] = {}
    for agent in sorted({r.expected_agent or r.called_agent or r.predicted_agent or "unknown" for r in results}):
        subset = [
            r for r in results
            if agent in {r.expected_agent, r.called_agent, r.predicted_agent}
        ]
        judged_subset = [r for r in subset if r.judge]
        by_agent[agent] = {
            "total": len(subset),
            "success": sum(1 for r in subset if not r.error),
            "errors": sum(1 for r in subset if r.error),
            "avg_keyword_hit_rate": (
                sum(r.keyword_hit_rate for r in subset) / len(subset)
                if subset else 0
            ),
            "judge_pass_rate": (
                sum(1 for r in judged_subset if bool(r.judge.get("pass"))) / len(judged_subset)
                if judged_subset else None
            ),
            "avg_latency_seconds": (
                sum(r.latency_seconds for r in subset) / len(subset)
                if subset else 0
            ),
        }

    by_type: dict[str, dict[str, Any]] = {}
    for result in results:
        prefix = result.case_id.split("_", 1)[0] if result.case_id else "unknown"
        by_type.setdefault(prefix, {"total": 0, "success": 0, "errors": 0})
        by_type[prefix]["total"] += 1
        if result.error:
            by_type[prefix]["errors"] += 1
        else:
            by_type[prefix]["success"] += 1

    return {
        "total": total,
        "success": len(ok),
        "errors": total - len(ok),
        "route_accuracy": (len(route_correct) / len(route_cases)) if route_cases else None,
        "avg_keyword_hit_rate": sum(r.keyword_hit_rate for r in results) / total if total else 0,
        "forbidden_hit_rate": sum(1 for r in results if r.forbidden_hit) / total if total else 0,
        "judge_pass_rate": (len(judged_pass) / len(judged)) if judged else None,
        "hallucination_rate": (len(hallucinations) / len(judged)) if judged else None,
        "avg_latency_seconds": sum(r.latency_seconds for r in results) / total if total else 0,
        "by_agent": by_agent,
        "by_type": by_type,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodel_RAG agents.")
    parser.add_argument("--cases", required=True, help="JSONL cases path")
    parser.add_argument("--out", default="eval/outputs/eval_results.jsonl")
    parser.add_argument("--report", default="eval/reports/eval_summary.json")
    parser.add_argument(
        "--mode",
        choices=["router", "expected_agent", "system"],
        default="expected_agent",
        help="router: route only; expected_agent: call case.expected_agent; system: call full System_Agent workflow",
    )
    parser.add_argument("--judge", action="store_true", help="Use an LLM judge")
    parser.add_argument(
        "--judge-provider",
        choices=["deepseek", "openai"],
        default=None,
        help="Override JUDGE_PROVIDER. Use openai for GPT-based judge.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override JUDGE_MODEL, e.g. gpt-5.5 or deepseek-chat.",
    )
    parser.add_argument(
        "--router-model",
        default=None,
        help="Override ROUTER_MODEL for router-only evaluation.",
    )
    parser.add_argument(
        "--only-agent",
        choices=["rag_agent", "graphrag_agent", "multi_model_agent"],
        default=None,
        help="Only evaluate cases whose expected_agent matches this agent.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.judge_provider:
        os.environ["JUDGE_PROVIDER"] = args.judge_provider
    if args.judge_model:
        os.environ["JUDGE_MODEL"] = args.judge_model
    if args.router_model:
        os.environ["ROUTER_MODEL"] = args.router_model

    cases = load_jsonl(Path(args.cases))
    if args.only_agent:
        cases = [case for case in cases if case.get("expected_agent") == args.only_agent]
    if args.limit:
        cases = cases[: args.limit]

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    results: list[EvalResult] = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case.get('id', '')} {case['question']}")
        result = await eval_one(case, args.mode, args.judge)
        results.append(result)
        append_jsonl(out_path, result.__dict__)
        if result.error:
            print("  ERROR:", result.error)
        else:
            print(
                "  agent=",
                result.called_agent or result.predicted_agent,
                "hit=",
                f"{result.keyword_hit_rate:.2f}",
                "pass=",
                result.judge.get("pass") if result.judge else "n/a",
            )

    summary = summarize(results)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {out_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
