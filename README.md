# rag-and-ragas

RAG 시스템을 설계하고 RAGAS로 평가하기 위한 작업 공간.

## `rag-design` 스킬

`.claude/skills/rag-design/` 에 RAG 설계 스킬이 있다. Claude Code에서 `/rag-design` 으로
부르거나, RAG 설계·진단 관련 요청을 하면 자동으로 걸린다.

무엇을 하는가:

1. 요구사항 인터뷰 (설계를 실제로 바꾸는 7개 질문)
2. 골든셋 확보 — **코드보다 먼저**
3. 베이스라인 파이프라인 확정 (청킹 / 임베딩 / 검색 / 리랭킹 / 생성)
4. RAGAS 평가 계획과 임계치
5. 설계 문서 산출
6. 증상 → 원인 진단표로 개선 루프

```
.claude/skills/rag-design/
├── SKILL.md                      설계 절차와 진단표
├── references/
│   ├── discovery.md              인터뷰 심화, 답변→설계 매핑
│   ├── ingestion.md              파싱, 청킹, 메타데이터, 인덱싱
│   ├── retrieval.md              임베딩, 하이브리드, 쿼리 변환, 리랭킹
│   ├── generation.md             프롬프트, 인용, 캐싱, 모델 선택
│   ├── evaluation.md             골든셋 제작, RAGAS 지표 해석
│   ├── patterns.md               아키텍처 패턴과 각각이 필요해지는 신호
│   └── operations.md             인덱스 갱신, 모니터링, 보안, 롤아웃
├── assets/
│   ├── design-doc-template.md    설계 문서 템플릿 (최종 산출물)
│   └── golden-set-sample.jsonl   골든셋 예시
└── scripts/
    └── ragas_eval.py             RAGAS 평가 하니스 (CI 게이트 가능)
```

## 평가 하니스

```bash
pip install ragas langchain-anthropic langchain-huggingface sentence-transformers

python .claude/skills/rag-design/scripts/ragas_eval.py \
  --golden data/golden.jsonl \
  --pipeline myapp.rag:answer \
  --out results/run.json
```

임계치(context_recall 0.85 / context_precision 0.70 / faithfulness 0.90 /
answer_relevancy 0.85) 미달이면 exit code 1을 반환한다.

RAGAS는 마이너 버전 사이에 API가 바뀐 이력이 있다. `ragas==x.y.z` 로 버전을 고정하고
설계 문서에 기록해 둘 것.
