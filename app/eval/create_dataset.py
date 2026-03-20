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
    # 1. 구어체 증상 표현 → search_symptoms
    # -----------------------------------------------------------------------
    {
        "input": "가슴이 답답해",
        "expected_output": (
            "가슴 답답함은 심장 질환, 역류성 식도염, 불안 장애, 폐 질환 등 다양한 원인일 수 있습니다. "
            "흉통이나 호흡곤란이 동반되면 즉시 응급실을 방문하세요."
        ),
        "category": "search_symptoms",
    },
    # -----------------------------------------------------------------------
    # 2. 비의료적 표현처럼 보이는 약물 질문 → 거절(non-medical)
    # -----------------------------------------------------------------------
    {
        "input": "아스피린 가격 얼마야",
        "expected_output": (
            "약물 가격 정보는 제공하지 않습니다. "
            "아스피린의 효능·용법·주의사항 등 의학 정보는 안내드릴 수 있습니다."
        ),
        "category": "non_medical",
    },
    # -----------------------------------------------------------------------
    # 3. 과장된 구어체 → search_symptoms + classify_emergency
    # -----------------------------------------------------------------------
    {
        "input": "머리가 깨질 것 같아, 눈도 침침하고 구역질 나",
        "expected_output": (
            "심한 두통과 구역질, 시력 변화가 동반되면 뇌압 상승이나 편두통, 뇌졸중 초기 증상일 수 있습니다. "
            "증상이 갑작스럽게 발생했다면 즉시 응급실을 방문하세요."
        ),
        "category": "classify_emergency",
    },
    # -----------------------------------------------------------------------
    # 4. 음식과 약 병용 → check_drug_interaction
    # -----------------------------------------------------------------------
    {
        "input": "타이레놀 먹고 소주 한 잔 해도 돼?",
        "expected_output": (
            "타이레놀(아세트아미노펜)과 알코올을 함께 섭취하면 간 독성 위험이 크게 증가합니다. "
            "음주 중이거나 음주 직후에는 복용을 피하세요."
        ),
        "category": "check_drug_interaction",
    },
    # -----------------------------------------------------------------------
    # 5. 복합 증상 조합 → classify_emergency (뇌수막염 의심)
    # -----------------------------------------------------------------------
    {
        "input": "열이 나고 목이 뻣뻣하고 빛이 너무 눈부셔서 못 뜨겠어",
        "expected_output": (
            "고열, 목 경직, 빛 과민 증상의 조합은 뇌수막염을 강하게 시사합니다. "
            "즉시 119에 신고하거나 응급실을 방문하세요."
        ),
        "category": "classify_emergency",
    },
    # -----------------------------------------------------------------------
    # 6. 일상어 증상 → search_symptoms
    # -----------------------------------------------------------------------
    {
        "input": "밥 먹고 나면 항상 배가 거북하고 트림이 많이 나",
        "expected_output": (
            "식후 더부룩함과 잦은 트림은 역류성 식도염, 기능성 소화불량 등의 증상일 수 있습니다. "
            "내과 또는 소화기내과 진료를 권장합니다."
        ),
        "category": "search_symptoms",
    },
    # -----------------------------------------------------------------------
    # 7. 심계항진 구어체 → classify_emergency
    # -----------------------------------------------------------------------
    {
        "input": "심장이 두근두근해서 잠을 못 자겠어. 이거 심각한 거야?",
        "expected_output": (
            "심계항진(두근거림)은 부정맥, 갑상선 이상, 빈혈 등 다양한 원인이 있을 수 있습니다. "
            "흉통·호흡곤란이 동반되면 즉시 응급실을 방문하세요."
        ),
        "category": "classify_emergency",
    },
    # -----------------------------------------------------------------------
    # 8. 복합 약물 질문 → check_drug_interaction
    # -----------------------------------------------------------------------
    {
        "input": "혈압약 먹는 중인데 감기약도 같이 먹어도 되나요?",
        "expected_output": (
            "혈압약과 감기약의 병용은 약물 종류에 따라 상호작용이 있을 수 있습니다. "
            "복용 중인 혈압약 이름을 확인하고 약사 또는 의사와 상담하세요."
        ),
        "category": "check_drug_interaction",
    },
    # -----------------------------------------------------------------------
    # 9. 병원 검색 (구어체 지역명) → find_nearby_hospitals
    # -----------------------------------------------------------------------
    {
        "input": "부산 쪽에 한의원 있으면 알려줘",
        "expected_output": (
            "부산 지역 한의원 목록을 조회합니다. "
            "예약 전 전화 확인을 권장합니다."
        ),
        "category": "find_nearby_hospitals",
    },
    # -----------------------------------------------------------------------
    # 10. 완전 비의료 질문 → 거절
    # -----------------------------------------------------------------------
    {
        "input": "오늘 날씨 어때?",
        "expected_output": (
            "죄송합니다, 저는 의료 정보 전문 AI 어시스턴트입니다. "
            "증상 검색, 약물 정보, 병원 찾기와 관련된 질문만 답변드릴 수 있습니다."
        ),
        "category": "non_medical",
    },
    # -----------------------------------------------------------------------
    # 11. 응급처치 구어체 → get_first_aid_guide
    # -----------------------------------------------------------------------
    {
        "input": "손목을 삐었는데 어떻게 해야 해?",
        "expected_output": (
            "손목 염좌 시 RICE 요법(휴식·냉찜질·압박·거상)을 적용하세요. "
            "심한 통증이나 뼈 골절이 의심되면 정형외과를 방문하세요."
        ),
        "category": "get_first_aid_guide",
    },
    # -----------------------------------------------------------------------
    # 12. 인지 기능 걱정 → search_symptoms
    # -----------------------------------------------------------------------
    {
        "input": "요즘 자꾸 깜빡하고 방에 왜 들어왔는지 모르겠어. 치매 초기인가?",
        "expected_output": (
            "건망증은 스트레스·수면 부족 등 일시적 원인일 수 있으나, "
            "치매 초기 증상과 구별이 필요합니다. 신경과 또는 정신건강의학과 전문의 상담을 권장합니다."
        ),
        "category": "search_symptoms",
    },
    # -----------------------------------------------------------------------
    # 13. 다리 부음 + 호흡곤란 복합 → classify_emergency (폐색전증 의심)
    # -----------------------------------------------------------------------
    {
        "input": "숨이 차고 다리가 부어올랐어. 그냥 지나쳐도 되나?",
        "expected_output": (
            "호흡곤란과 하지 부종의 조합은 심부전, 폐색전증 등 중증 질환의 신호일 수 있습니다. "
            "즉시 응급실을 방문하거나 119에 신고하세요."
        ),
        "category": "classify_emergency",
    },
    # -----------------------------------------------------------------------
    # 14. 소화제 추천 질문 → get_medication_info
    # -----------------------------------------------------------------------
    {
        "input": "속이 너무 안 좋은데 소화제 뭐가 좋아?",
        "expected_output": (
            "소화제는 증상에 따라 종류가 다릅니다. "
            "더부룩함에는 시메티콘 계열, 소화 불량에는 소화 효소 제제가 도움이 될 수 있습니다. "
            "지속적인 증상은 내과 진료를 권장합니다."
        ),
        "category": "get_medication_info",
    },
    # -----------------------------------------------------------------------
    # 15. 복합 도구 질문 (약물 + 병원) → get_medication_info + find_nearby_hospitals
    # -----------------------------------------------------------------------
    {
        "input": "이부프로펜 정보 알려주고, 대전 내과도 찾아줘",
        "expected_output": (
            "이부프로펜은 해열·진통·항염 효과가 있으며 공복 복용을 피해야 합니다. "
            "대전 지역 내과 병원 목록도 함께 안내드립니다."
        ),
        "category": "multi_tool",
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
