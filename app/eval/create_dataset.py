"""
Opik 데이터셋 생성 스크립트

사용법:
    uv run python -m app.eval.create_dataset
    uv run python -m app.eval.create_dataset --dataset my-dataset --overwrite
"""

from __future__ import annotations

import argparse
import os

from app.core.config import settings


def _configure_opik_env() -> None:
    if settings.OPIK is not None:
        opik_cfg = settings.OPIK
        if opik_cfg.URL_OVERRIDE:
            os.environ["OPIK_URL_OVERRIDE"] = opik_cfg.URL_OVERRIDE
        if opik_cfg.API_KEY:
            os.environ["OPIK_API_KEY"] = opik_cfg.API_KEY
        if opik_cfg.WORKSPACE:
            os.environ["OPIK_WORKSPACE"] = opik_cfg.WORKSPACE
        if opik_cfg.PROJECT:
            os.environ["OPIK_PROJECT_NAME"] = opik_cfg.PROJECT

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


# ---------------------------------------------------------------------------
# 데이터셋 아이템
# - input: 사용자 질문
# - expected_output: 에이전트가 반환해야 할 이상적인 답변 (평가 기준)
# ---------------------------------------------------------------------------

DATASET_ITEMS = [
    # -----------------------------------------------------------------------
    # 1. search_symptoms: 증상 기반 의료 정보 검색
    # -----------------------------------------------------------------------
    {
        "input": "두통이 심하고 구역질이 나요. 어떤 질환일 수 있나요?",
        "expected_output": (
            "두통과 구역질은 편두통, 뇌압 상승, 위장 장애 등 다양한 원인으로 나타날 수 있습니다. "
            "증상이 심하거나 지속되면 신경과 또는 내과를 방문하시기 바랍니다."
        ),
        "category": "search_symptoms",
    },
    # -----------------------------------------------------------------------
    # 2. get_medication_info: 약물 정보 조회
    # -----------------------------------------------------------------------
    {
        "input": "타이레놀 복용 방법과 주의사항을 알려줘.",
        "expected_output": (
            "타이레놀(아세트아미노펜)은 성인 기준 1회 500mg~1000mg, 하루 최대 4000mg 이하로 복용합니다. "
            "음주 중이거나 간 질환이 있는 경우 복용을 피하고, 다른 해열진통제와 병용 시 주의하세요."
        ),
        "category": "get_medication_info",
    },
    {
        "input": "이부프로펜은 어떤 효능이 있고 부작용은 무엇인가요?",
        "expected_output": (
            "이부프로펜은 해열, 진통, 항염증 효과가 있습니다. "
            "위장 장애, 소화불량, 드물게 위출혈이 부작용으로 나타날 수 있으며, 공복 복용은 피하세요."
        ),
        "category": "get_medication_info",
    },
    # -----------------------------------------------------------------------
    # 3. find_nearby_hospitals: 병원 검색
    # -----------------------------------------------------------------------
    {
        "input": "서울에 있는 정형외과 병원 찾아줘.",
        "expected_output": (
            "서울 지역 정형외과 병원 목록을 조회합니다. "
            "예약 전 전화 확인을 권장하며, 응급 상황에는 119에 연락하세요."
        ),
        "category": "find_nearby_hospitals",
    },
    # -----------------------------------------------------------------------
    # 4. check_drug_interaction: 병용금기 조회
    # -----------------------------------------------------------------------
    {
        "input": "아스피린과 와파린을 함께 먹어도 되나요?",
        "expected_output": (
            "아스피린과 와파린은 병용 시 출혈 위험이 크게 증가합니다. "
            "두 약물을 함께 복용해야 하는 경우 반드시 의사와 상담 후 결정하세요."
        ),
        "category": "check_drug_interaction",
    },
    {
        "input": "타이레놀이랑 이부프로펜 같이 먹으면 어떻게 되나요?",
        "expected_output": (
            "타이레놀(아세트아미노펜)과 이부프로펜은 작용 기전이 달라 병용 가능한 경우도 있으나, "
            "의사 또는 약사 확인 없이 임의로 병용하는 것은 피하세요."
        ),
        "category": "check_drug_interaction",
    },
    # -----------------------------------------------------------------------
    # 5. classify_emergency: 응급 여부 판단
    # -----------------------------------------------------------------------
    {
        "input": "갑자기 가슴을 쥐어짜는 심한 흉통이 오고 왼팔이 저려요. 응급인가요?",
        "expected_output": (
            "심장마비 증상일 수 있습니다. 즉시 119에 신고하세요. "
            "심한 흉통과 왼팔 저림은 급성 심근경색의 주요 증상으로 생명을 위협할 수 있습니다."
        ),
        "category": "classify_emergency",
    },
    {
        "input": "고열이 39도가 넘고 호흡이 빠른데 응급실에 가야 하나요?",
        "expected_output": (
            "39도 이상 고열과 빠른 호흡은 폐렴이나 패혈증 등을 의심할 수 있어 응급실 방문이 필요합니다. "
            "증상이 급격히 악화되면 즉시 119에 신고하세요."
        ),
        "category": "classify_emergency",
    },
    # -----------------------------------------------------------------------
    # 6. get_first_aid_guide: 응급처치 안내
    # -----------------------------------------------------------------------
    {
        "input": "뜨거운 물에 손을 데었어요. 응급처치 어떻게 해야 하나요?",
        "expected_output": (
            "즉시 흐르는 찬물로 15~20분 이상 화상 부위를 식히세요. "
            "얼음 직접 사용, 물집 터뜨리기, 된장이나 치약 바르기는 금지입니다. "
            "범위가 넓거나 심한 경우 응급실을 방문하세요."
        ),
        "category": "get_first_aid_guide",
    },
    {
        "input": "넘어져서 팔뼈가 부러진 것 같아요. 어떻게 해야 하나요?",
        "expected_output": (
            "골절 부위를 움직이지 말고 부목으로 고정하세요. "
            "개방성 골절이면 깨끗한 천으로 덮고 뼈를 밀어 넣지 마세요. "
            "즉시 119에 신고하거나 응급실을 방문하세요."
        ),
        "category": "get_first_aid_guide",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Opik 데이터셋 생성")
    parser.add_argument(
        "--dataset",
        default="yja-dataset",
        help="생성할 데이터셋 이름 (default: yja-dataset)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="데이터셋이 이미 존재할 경우 기존 아이템을 삭제하고 새로 삽입",
    )
    args = parser.parse_args()

    _configure_opik_env()

    import opik

    client = opik.Opik()
    dataset = client.get_or_create_dataset(name=args.dataset)

    if args.overwrite:
        existing = dataset.get_items()
        if existing:
            ids = [item.id if hasattr(item, "id") else item["id"] for item in existing]
            dataset.delete(ids)
            print(f"기존 아이템 {len(ids)}개 삭제 완료")

    dataset.insert(DATASET_ITEMS)
    print(f"데이터셋 '{args.dataset}'에 {len(DATASET_ITEMS)}개 아이템 삽입 완료")
    print("Opik UI에서 확인: Datasets > " + args.dataset)


if __name__ == "__main__":
    main()
