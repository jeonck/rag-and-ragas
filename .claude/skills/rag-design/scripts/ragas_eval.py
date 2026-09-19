#!/usr/bin/env python3
"""RAG 파이프라인을 골든셋으로 평가하는 하니스.

골든셋(JSONL)과 파이프라인 함수를 받아 RAGAS 4지표를 내고, 임계치 미달이면
exit code 1을 반환한다. CI 게이트로 그대로 쓸 수 있다.

사용:
    # 파이프라인을 직접 호출해서 평가
    python ragas_eval.py --golden data/golden.jsonl --pipeline myapp.rag:answer

    # 이미 만들어 둔 예측 결과로 평가
    python ragas_eval.py --golden data/golden.jsonl --predictions runs/preds.jsonl

골든셋 한 줄:
    {"question": "...", "ground_truth": "...", "tags": ["single-hop"]}

파이프라인 함수 규약:
    def answer(question: str) -> dict:   # {"answer": str, "contexts": list[str]}

RAGAS는 마이너 버전 사이에 API가 바뀐 이력이 있다. 이 스크립트는 신/구 API를
모두 시도하지만, 설치 버전을 핀으로 고정하고(`ragas==x.y.z`) 설계 문서에
적어두는 편이 안전하다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 초기 목표치. 프로젝트에 맞게 --thresholds 로 덮어쓴다.
DEFAULT_THRESHOLDS = {
    "context_recall": 0.85,
    "context_precision": 0.70,
    "faithfulness": 0.90,
    "answer_relevancy": 0.85,
}

CANONICAL = list(DEFAULT_THRESHOLDS)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} JSON 파싱 실패: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: 비어 있다")
    return rows


def resolve_pipeline(spec: str):
    """'myapp.rag:answer' -> 호출 가능한 함수."""
    if ":" not in spec:
        raise SystemExit("--pipeline 은 '모듈:함수' 형식이어야 한다 (예: myapp.rag:answer)")
    module_name, func_name = spec.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"모듈 '{module_name}' 을 import 할 수 없다: {exc}") from exc
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise SystemExit(f"'{module_name}' 에 '{func_name}' 가 없다") from exc


def run_pipeline(fn, golden: list[dict], concurrency: int) -> list[dict]:
    """골든셋 질문을 파이프라인에 태워 예측을 만든다."""

    def one(item: dict) -> dict:
        out = fn(item["question"])
        if not isinstance(out, dict) or "answer" not in out or "contexts" not in out:
            raise SystemExit(
                "파이프라인 함수는 {'answer': str, 'contexts': list[str]} 를 반환해야 한다. "
                f"받은 것: {type(out).__name__}"
            )
        return {**item, "answer": out["answer"], "contexts": list(out["contexts"])}

    if concurrency <= 1:
        results = []
        for i, item in enumerate(golden, 1):
            print(f"  [{i}/{len(golden)}] {item['question'][:60]}", file=sys.stderr)
            results.append(one(item))
        return results

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, golden))


def build_judges(judge_model: str, embedding_model: str):
    """RAGAS 판정용 LLM과 임베딩. 둘 다 고정해야 점수를 비교할 수 있다."""
    try:
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise SystemExit(
            "판정 모델 준비 실패. `pip install ragas langchain-anthropic` 후 "
            "ANTHROPIC_API_KEY 를 설정한다.\n" f"  원인: {exc}"
        ) from exc

    llm = LangchainLLMWrapper(ChatAnthropic(model=judge_model, max_tokens=4096))

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model))
    except ImportError:
        print(
            "경고: 임베딩 모델을 불러오지 못했다. answer_relevancy 가 빠질 수 있다.\n"
            "  `pip install langchain-huggingface sentence-transformers` 로 설치한다.",
            file=sys.stderr,
        )
        emb = None

    return llm, emb


def evaluate_modern(preds: list[dict], llm, emb):
    """ragas 0.2+ 경로."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": p["question"],
                "retrieved_contexts": p["contexts"],
                "response": p["answer"],
                "reference": p.get("ground_truth", ""),
            }
            for p in preds
        ]
    )

    metrics = [LLMContextRecall(), LLMContextPrecisionWithReference(), Faithfulness()]
    if emb is not None:
        metrics.append(ResponseRelevancy())

    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb)


def evaluate_legacy(preds: list[dict], llm, emb):
    """ragas 0.1.x 경로."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [p["question"] for p in preds],
            "contexts": [p["contexts"] for p in preds],
            "answer": [p["answer"] for p in preds],
            "ground_truth": [p.get("ground_truth", "") for p in preds],
        }
    )
    metrics = [context_recall, context_precision, faithfulness]
    if emb is not None:
        metrics.append(answer_relevancy)

    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb)


def to_scores(result) -> dict[str, float]:
    """RAGAS 결과에서 지표명 -> 평균점수. 버전마다 컬럼명이 달라 부분 일치로 맞춘다."""
    raw: dict[str, float] = {}
    try:
        df = result.to_pandas()
        for col in df.columns:
            series = df[col]
            if series.dtype.kind in "fi":
                raw[col] = float(series.mean())
    except Exception:
        try:
            raw = {k: float(v) for k, v in dict(result).items()}
        except Exception as exc:
            raise SystemExit(f"RAGAS 결과를 읽지 못했다: {exc}") from exc

    scores: dict[str, float] = {}
    for canon in CANONICAL:
        for key, value in raw.items():
            normalized = key.lower().replace("-", "_")
            if canon in normalized or normalized in canon:
                scores[canon] = value
                break
    # 매핑되지 않은 지표도 참고용으로 남긴다.
    for key, value in raw.items():
        scores.setdefault(key, value)
    return scores


def report(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
    print("\n" + "=" * 56)
    print(f"{'지표':<28}{'점수':>8}{'임계치':>10}{'':>6}")
    print("-" * 56)
    passed = True
    for name in CANONICAL:
        if name not in scores:
            print(f"{name:<28}{'—':>8}{'':>10}  (측정 안 됨)")
            continue
        score = scores[name]
        limit = thresholds.get(name)
        if limit is None:
            print(f"{name:<28}{score:>8.3f}{'—':>10}")
            continue
        ok = score >= limit
        passed = passed and ok
        print(f"{name:<28}{score:>8.3f}{limit:>10.2f}{'  PASS' if ok else '  FAIL':>6}")
    extras = {k: v for k, v in scores.items() if k not in CANONICAL}
    if extras:
        print("-" * 56)
        for name, score in extras.items():
            print(f"{name:<28}{score:>8.3f}")
    print("=" * 56)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 파이프라인 RAGAS 평가")
    parser.add_argument("--golden", type=Path, required=True, help="골든셋 JSONL")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pipeline", help="'모듈:함수' 형식의 파이프라인 진입점")
    src.add_argument("--predictions", type=Path, help="예측 결과 JSONL (question/answer/contexts)")
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    parser.add_argument("--judge-model", default="claude-opus-5", help="RAGAS 판정 모델 (고정할 것)")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3", help="판정용 임베딩 모델")
    parser.add_argument("--concurrency", type=int, default=1, help="파이프라인 병렬 실행 수")
    parser.add_argument("--limit", type=int, help="앞에서 N개만 평가 (빠른 확인용)")
    parser.add_argument("--tag", help="이 태그가 붙은 문항만 평가")
    parser.add_argument(
        "--thresholds",
        help="임계치 덮어쓰기. 예: context_recall=0.8,faithfulness=0.95",
    )
    args = parser.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        for pair in args.thresholds.split(","):
            key, _, value = pair.partition("=")
            thresholds[key.strip()] = float(value)

    golden = load_jsonl(args.golden)
    if args.tag:
        golden = [g for g in golden if args.tag in g.get("tags", [])]
        if not golden:
            raise SystemExit(f"태그 '{args.tag}' 에 해당하는 문항이 없다")
    if args.limit:
        golden = golden[: args.limit]

    if args.predictions:
        by_question = {p["question"]: p for p in load_jsonl(args.predictions)}
        missing = [g["question"] for g in golden if g["question"] not in by_question]
        if missing:
            raise SystemExit(f"예측이 없는 문항 {len(missing)}개. 첫 번째: {missing[0][:60]}")
        preds = [{**g, **by_question[g["question"]]} for g in golden]
    else:
        print(f"파이프라인 실행: {len(golden)}문항", file=sys.stderr)
        preds = run_pipeline(resolve_pipeline(args.pipeline), golden, args.concurrency)

    empty = sum(1 for p in preds if not p["contexts"])
    if empty:
        print(f"경고: 검색 결과가 빈 문항 {empty}개 — 검색 단계를 먼저 확인한다", file=sys.stderr)

    llm, emb = build_judges(args.judge_model, args.embedding_model)

    print(f"RAGAS 평가 중 (판정 모델: {args.judge_model})...", file=sys.stderr)
    try:
        result = evaluate_modern(preds, llm, emb)
    except ImportError:
        print("  최신 RAGAS API 없음 — 구 API로 재시도", file=sys.stderr)
        result = evaluate_legacy(preds, llm, emb)

    scores = to_scores(result)
    passed = report(scores, thresholds)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "scores": scores,
                    "thresholds": thresholds,
                    "passed": passed,
                    "n_items": len(preds),
                    "judge_model": args.judge_model,
                    "embedding_model": args.embedding_model,
                    "tag": args.tag,
                    "per_item": [
                        {
                            "question": p["question"],
                            "answer": p["answer"],
                            "n_contexts": len(p["contexts"]),
                            "tags": p.get("tags", []),
                        }
                        for p in preds
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"결과 저장: {args.out}", file=sys.stderr)

    if not passed:
        print("임계치 미달 — 진단표는 SKILL.md 참고", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
