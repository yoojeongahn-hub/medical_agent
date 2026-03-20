"""
Opik evaluation 실행 스크립트

사용법:
    # 기본 실행 (AnswerRelevance + Hallucination)
    uv run python -m app.eval.run_opik_eval

    # 실험 이름 지정
    uv run python -m app.eval.run_opik_eval --experiment gpt-4o-v1

    # 샘플 수 제한 + 로그 최소화
    uv run python -m app.eval.run_opik_eval --nb-samples 5 --quiet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
import logging
import warnings
from typing import Any, Iterable

from app.core.config import settings
from app.services.agent_service import AgentService


def _configure_opik_env() -> None:
    if settings.OPIK is not None:
        opik_settings = settings.OPIK
        if opik_settings.URL_OVERRIDE:
            os.environ["OPIK_URL_OVERRIDE"] = opik_settings.URL_OVERRIDE
        if opik_settings.API_KEY:
            os.environ["OPIK_API_KEY"] = opik_settings.API_KEY
        if opik_settings.WORKSPACE:
            os.environ["OPIK_WORKSPACE"] = opik_settings.WORKSPACE
        if opik_settings.PROJECT:
            os.environ["OPIK_PROJECT_NAME"] = opik_settings.PROJECT

    nested_map = {
        "OPIK__URL_OVERRIDE": "OPIK_URL_OVERRIDE",
        "OPIK__API_KEY": "OPIK_API_KEY",
        "OPIK__WORKSPACE": "OPIK_WORKSPACE",
        "OPIK__PROJECT": "OPIK_PROJECT_NAME",
    }
    for nested_key, flat_key in nested_map.items():
        if os.environ.get(flat_key):
            continue
        value = os.environ.get(nested_key)
        if value:
            os.environ[flat_key] = value


def _first_present(d: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


@dataclass(frozen=True)
class DatasetFieldHints:
    input_keys: tuple[str, ...] = (
        "input",
        "question",
        "query",
        "prompt",
        "message",
        "user_message",
        "user",
        "content",
        "text",
    )
    reference_keys: tuple[str, ...] = (
        "reference",
        "expected",
        "expected_output",
        "expected_answer",
        "answer",
        "ground_truth",
        "label",
    )


async def _run_agent_once(agent_service: AgentService, user_message: str) -> str:
    """에이전트를 한 번 실행하고 최종 응답(done 페이로드)을 반환합니다.

    스트림을 끝까지 소비해야 LangGraph 클린업(체크포인트 저장, Opik 트레이스 완료)이
    정상적으로 실행됩니다. 중간에 return하면 제너레이터가 버려져 output이 기록되지 않습니다.
    """
    thread_id = uuid.uuid4()
    result = ""
    async for chunk in agent_service.process_query(user_messages=user_message, thread_id=thread_id):
        try:
            payload = json.loads(chunk)
        except Exception:
            continue
        if payload.get("step") == "done":
            result = _coerce_str(payload.get("content"))
    return result


async def _collect_outputs(
    items: list[Any],
    hints: DatasetFieldHints,
) -> dict[str, dict[str, str]]:
    """
    데이터셋 아이템 전체를 단일 이벤트 루프에서 순차 실행하고
    { user_msg -> {input, output, reference} } 형태로 반환합니다.

    asyncio.run()을 task()마다 호출하면 이벤트 루프가 매번 새로 생성되어
    aiosqlite 연결이 충돌합니다. 이를 방지하기 위해 모든 추론을 여기서 일괄 실행합니다.
    """
    agent_service = AgentService()
    results: dict[str, dict[str, str]] = {}

    for item in items:
        data: dict[str, Any] = item.get_content() if hasattr(item, "get_content") else item
        user_msg = _coerce_str(_first_present(data, hints.input_keys))
        reference = _coerce_str(_first_present(data, hints.reference_keys))

        print(f"  실행 중: {user_msg[:60]}...")
        output = await _run_agent_once(agent_service, user_msg)
        results[user_msg] = {"input": user_msg, "output": output, "reference": reference}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Opik evaluation 실행")
    parser.add_argument("--dataset", default="yja-dataset", help="Opik 데이터셋 이름 (default: yja-dataset)")
    parser.add_argument("--experiment", default=None, help="실험 이름 (default: 자동 생성)")
    parser.add_argument("--project", default=None, help="Opik 프로젝트 이름 (default: .env 설정값)")
    parser.add_argument("--nb-samples", type=int, default=None, help="평가할 샘플 수 (default: 전체)")
    parser.add_argument("--threads", type=int, default=1, help="task 병렬 스레드 수 (default: 1)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "answer_relevance",
            "hallucination",
            "rouge",
            "bertscore",
            "meteor",
            "moderation",
            "usefulness",
            "task_completion",
            "tool_correctness",
        ],
        default=["answer_relevance", "hallucination"],
        help=(
            "사용할 메트릭 (default: answer_relevance hallucination)\n"
            "  answer_relevance  : 질문 대비 응답 관련성 (LLM)\n"
            "  hallucination     : 환각 여부 (LLM)\n"
            "  rouge             : 단어 겹침 기반 유사도 (reference 필요)\n"
            "  bertscore         : 의미 기반 유사도 (reference 필요)\n"
            "  meteor            : 어간/동의어 포함 유사도 (reference 필요)\n"
            "  moderation        : 유해 콘텐츠 여부 (LLM)\n"
            "  usefulness        : 응답 유용성 (LLM)\n"
            "  task_completion   : 태스크 완료도 (LLM)\n"
            "  tool_correctness  : 도구 선택 적절성 (LLM)\n"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="로그/진행상황 출력 최소화")
    args = parser.parse_args()

    _configure_opik_env()

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("edu_agent").setLevel(logging.ERROR)
        logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)
        logging.getLogger("elastic_transport.transport").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        os.environ.setdefault("OPIK_CONSOLE_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("OPIK_LOG_START_TRACE_SPAN", "false")

    import opik
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import (
        AnswerRelevance,
        Hallucination,
        ROUGE,
        BERTScore,
        METEOR,
        Moderation,
        Usefulness,
        AgentTaskCompletionJudge,
        AgentToolCorrectnessJudge,
    )

    # ---------------------------------------------------------------------------
    # 메트릭 구성
    # ---------------------------------------------------------------------------
    scoring_metrics = []
    if "answer_relevance" in args.metrics:
        # 질문 대비 응답 관련성 (0~1, 높을수록 좋음)
        # require_context=False: context 없이도 input/output만으로 평가
        scoring_metrics.append(AnswerRelevance(require_context=False))
    if "hallucination" in args.metrics:
        # 환각(사실과 다른 정보) 포함 여부 (0~1, 낮을수록 좋음)
        scoring_metrics.append(Hallucination())
    if "rouge" in args.metrics:
        # 단어 겹침 기반 유사도 - reference와 비교 (0~1, 높을수록 좋음)
        scoring_metrics.append(ROUGE())
    if "bertscore" in args.metrics:
        # 의미 기반 유사도 - reference와 비교 (0~1, 높을수록 좋음)
        scoring_metrics.append(BERTScore())
    if "meteor" in args.metrics:
        # 어간/동의어 포함 유사도 - reference와 비교 (0~1, 높을수록 좋음)
        scoring_metrics.append(METEOR())
    if "moderation" in args.metrics:
        # 유해하거나 위험한 의료 정보 포함 여부 (0~1, 낮을수록 좋음)
        scoring_metrics.append(Moderation())
    if "usefulness" in args.metrics:
        # 응답이 실제로 유용한지 (0~1, 높을수록 좋음)
        scoring_metrics.append(Usefulness())
    if "task_completion" in args.metrics:
        # 태스크를 끝까지 완료했는지 (0~1, 높을수록 좋음)
        scoring_metrics.append(AgentTaskCompletionJudge())
    if "tool_correctness" in args.metrics:
        # 올바른 도구를 선택했는지 (0~1, 높을수록 좋음)
        scoring_metrics.append(AgentToolCorrectnessJudge())

    # ---------------------------------------------------------------------------
    # 데이터셋 로드
    # ---------------------------------------------------------------------------
    project_name = args.project or (settings.OPIK.PROJECT if settings.OPIK else None)
    client = opik.Opik(project_name=project_name)
    dataset = client.get_dataset(args.dataset)

    all_items = dataset.get_items()
    if args.nb_samples:
        all_items = all_items[: args.nb_samples]

    if not all_items:
        print(f"데이터셋 '{args.dataset}'에 아이템이 없습니다. create_dataset.py를 먼저 실행하세요.")
        return

    print(f"총 {len(all_items)}개 아이템에 대해 에이전트를 실행합니다...")

    # ---------------------------------------------------------------------------
    # 에이전트 추론: 단일 이벤트 루프에서 모든 아이템을 순차 처리
    # ---------------------------------------------------------------------------
    hints = DatasetFieldHints()
    precomputed = asyncio.run(_collect_outputs(all_items, hints))

    print(f"\n추론 완료. Opik evaluation 시작...\n")

    # ---------------------------------------------------------------------------
    # Opik evaluate: task()는 precomputed 결과를 반환하기만 함
    # ---------------------------------------------------------------------------
    def task(dataset_item: dict[str, Any]) -> dict[str, Any]:
        user_msg = _coerce_str(_first_present(dataset_item, hints.input_keys))
        return precomputed.get(
            user_msg,
            {"input": user_msg, "output": "", "reference": ""},
        )

    result = evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=scoring_metrics,
        experiment_name=args.experiment,
        project_name=project_name,
        nb_samples=args.nb_samples,
        task_threads=args.threads,
        experiment_config={
            "openai_model": settings.OPENAI_MODEL,
            "dataset": args.dataset,
            "metrics": args.metrics,
        },
        verbose=0 if args.quiet else 1,
    )

    if not args.quiet:
        print(result)


if __name__ == "__main__":
    main()

