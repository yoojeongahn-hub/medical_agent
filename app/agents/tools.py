from __future__ import annotations

from typing import Any

import httpx
from elasticsearch import Elasticsearch
from langchain.tools import tool
from langchain_elasticsearch import ElasticsearchRetriever

# ---------------------------------------------------------------------------
# Elasticsearch 연결 설정
# ---------------------------------------------------------------------------

_ES_URL = "https://elasticsearch-edu.didim365.app"
_ES_USER = "elastic"
_ES_PASSWORD = "FJl79PA7mMIJajxB1OHgdLEe"
_INDEX_NAME = "edu-collection"
_CONTENT_FIELD = "content"
_TOP_K = 5

# 식품의약품안전처(e약은요) API
_DRUG_API_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
_DRUG_API_KEY = "72c6779aea30770c96a2620ae1c96d6acb8ab33c5c1fdb404b91ac8864927e0e"

# 식품의약품안전처 DUR 병용금기 API
_DUR_API_URL = "https://apis.data.go.kr/1471000/DURInfrService01/getUsjntTabooInfoList01"
_DUR_API_KEY = _DRUG_API_KEY

# 건강보험심사평가원 병원정보서비스 API
_HOSP_API_URL = "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"
_HOSP_API_KEY = "72c6779aea30770c96a2620ae1c96d6acb8ab33c5c1fdb404b91ac8864927e0e"

# 시도명 → sidoCd 매핑 (건강보험심사평가원 코드 기준)
_SIDO_CODE: dict[str, str] = {
    "서울": "110000", "서울특별시": "110000",
    "부산": "210000", "부산광역시": "210000",
    "대구": "220000", "대구광역시": "220000",
    "인천": "230000", "인천광역시": "230000",
    "광주": "240000", "광주광역시": "240000",
    "대전": "250000", "대전광역시": "250000",
    "울산": "260000", "울산광역시": "260000",
    "세종": "290000", "세종특별자치시": "290000",
    "경기": "410000", "경기도": "410000",
    "강원": "420000", "강원도": "420000", "강원특별자치도": "420000",
    "충북": "430000", "충청북도": "430000",
    "충남": "440000", "충청남도": "440000",
    "전북": "450000", "전라북도": "450000", "전북특별자치도": "450000",
    "전남": "460000", "전라남도": "460000",
    "경북": "470000", "경상북도": "470000",
    "경남": "480000", "경상남도": "480000",
    "제주": "490000", "제주특별자치도": "490000",
}

# 병원 종별 → clCd 매핑 (건강보험심사평가원 종별코드)
_CL_CODE: dict[str, str] = {
    "상급종합": "01", "상급종합병원": "01",
    "종합병원": "11",
    "병원": "21",
    "요양병원": "28", "정신병원": "29",
    "의원": "31", "일반": "31",
    "치과": "41", "치과병원": "41", "치과의원": "42",
    "한방병원": "51", "한의원": "52",
    "보건소": "61", "보건지소": "62",
    "약국": "92",
}

# 진료과목명 → dgsbjtCd 매핑 (건강보험심사평가원 진료과목코드)
_DEPT_CODE: dict[str, str] = {
    "일반의": "00",
    "내과": "01",
    "신경과": "02",
    "정신건강의학과": "03", "정신과": "03",
    "외과": "04",
    "정형외과": "05",
    "신경외과": "06",
    "심장혈관흉부외과": "07", "흉부외과": "07",
    "성형외과": "08",
    "마취통증의학과": "09", "마취과": "09",
    "산부인과": "10", "산부과": "10",
    "소아청소년과": "11", "소아과": "11",
    "안과": "12",
    "이비인후과": "13",
    "피부과": "14",
    "비뇨의학과": "15", "비뇨과": "15",
    "영상의학과": "16",
    "방사선종양학과": "17",
    "병리과": "18",
    "진단검사의학과": "19",
    "결핵과": "20",
    "재활의학과": "21", "재활과": "21",
    "핵의학과": "22",
    "가정의학과": "23", "가정과": "23",
    "응급의학과": "24", "응급과": "24",
    "직업환경의학과": "25",
    "예방의학과": "26",
    # '치과'는 clCd=41(치과병원)으로 처리하므로 _DEPT_CODE에서 제외
    "한방": "28",
    "한방내과": "80",
    "한방부인과": "81",
    "한방소아과": "82",
    "침구과": "85",
    "한방재활의학과": "86",
    "사상체질과": "87",
}


def _bm25_query(search_query: str) -> dict[str, Any]:
    """BM25 match 쿼리 빌더 함수 (ElasticsearchRetriever body_func 구조)"""
    return {
        "query": {
            "match": {
                _CONTENT_FIELD: {
                    "query": search_query,
                    "operator": "or",
                }
            }
        },
        "size": _TOP_K,
    }


def _build_retriever() -> ElasticsearchRetriever:
    """Elasticsearch 클라이언트를 생성한 뒤 ElasticsearchRetriever에 주입합니다."""
    es_client = Elasticsearch(
        _ES_URL,
        basic_auth=(_ES_USER, _ES_PASSWORD),
        verify_certs=False,
    )
    return ElasticsearchRetriever(
        index_name=_INDEX_NAME,
        body_func=_bm25_query,
        content_field=_CONTENT_FIELD,
        client=es_client,
    )


# 전역 싱글턴 인스턴스 (카단 황유)
_retriever: ElasticsearchRetriever | None = None


def _get_retriever() -> ElasticsearchRetriever:
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever()
    return _retriever


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_symptoms(symptoms: str) -> str:
    """주어진 증상(쉼표로 구분)을 기반으로 Elasticsearch에서 관련 의료 정보를 검색합니다."""
    retriever = _get_retriever()
    docs = retriever.invoke(symptoms)
    if not docs:
        return f"증상 '{symptoms}'에 대한 관련 의료 정보를 찾을 수 없습니다."

    results: list[str] = []
    for i, doc in enumerate(docs, 1):
        source_spec = doc.metadata.get("_source", {}).get("source_spec", "unknown")
        creation_year = doc.metadata.get("_source", {}).get("creation_year", "")
        header = f"[{i}] 출처: {source_spec}" + (f" ({creation_year}년)" if creation_year and creation_year != "null" else "")
        snippet = doc.page_content[:500].replace("\n", " ")
        results.append(f"{header}\n{snippet}")

    return "\n\n".join(results)

# 영어 약이름 → 한국어 검색어 매핑 (e약은요 API는 한국어 제품명 기반)
_DRUG_NAME_ALIASES: dict[str, str] = {
    "aspirin": "아스피린",
    "acetaminophen": "아세트아미노펜",
    "tylenol": "타이레놀",
    "ibuprofen": "이부프로펜",
    "advil": "이부프로펜",
    "amoxicillin": "아목시실린",
    "metformin": "메트포르민",
    "warfarin": "와파린",
    "amlodipine": "암로디핀",
    "atorvastatin": "아토르바스타틴",
    "omeprazole": "오메프라졸",
    "losartan": "로사르탄",
}


def _strip_html(text: str) -> str:
    """API 응답에서 HTML 태그를 제거합니다."""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()


@tool
def get_medication_info(medication_name: str) -> str:
    """약물 이름을 받아 식품의약품안전처(e약은요) API에서 효능, 사용법, 주의사항, 부작용, 보관법 등을 조회합니다."""
    # 영어 약이름이면 한국어로 변환
    search_name = _DRUG_NAME_ALIASES.get(medication_name.lower().strip(), medication_name.strip())

    try:
        resp = httpx.get(
            _DRUG_API_URL,
            params={
                "serviceKey": _DRUG_API_KEY,
                "itemName": search_name,
                "type": "json",
                "numOfRows": 3,
                "pageNo": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return f"약물 정보 API 호출 실패 (HTTP {e.response.status_code}): {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"약물 정보 API 네트워크 오류: {e}"
    except Exception as e:
        return f"약물 정보 조회 중 오류 발생: {e}"

    items: list[Any] = (
        data.get("body", {}).get("items", [])
        or data.get("response", {}).get("body", {}).get("items", [])
        or []
    )
    if isinstance(items, dict):
        items = [items]

    if not items:
        return f"'{medication_name}'에 대한 의약품 정보를 찾을 수 없습니다."

    lines: list[str] = []
    for item in items:
        name = item.get("itemName", search_name)
        entp = item.get("entpName", "")
        lines.append(f"■ 제품명: {name}" + (f" ({entp})" if entp else ""))

        field_map = {
            "efcyQesitm": "효능",
            "useMethodQesitm": "사용법",
            "atpnWarnQesitm": "주의사항(경고)",
            "atpnQesitm": "주의사항",
            "intrcQesitm": "상호작용",
            "seQesitm": "부작용",
            "depositMethodQesitm": "보관법",
        }
        for field, label in field_map.items():
            value = item.get(field)
            if value:
                lines.append(f"  [{label}] {_strip_html(value)}")
        lines.append("")

    return "\n".join(lines).strip()

_EMERGENCY_KEYWORDS: dict[str, list[str]] = {
    "즉시 119 신고 (생명위협)": [
        "심정지", "심장마비", "호흡정지", "의식없음", "의식불명", "쓰러짐", "쓰러졌",
        "뇌졸중", "뇌출혈", "뇌경색", "반신마비", "안면마비", "갑작스러운 마비",
        "대량출혈", "지혈안됨", "토혈", "혈변 다량", "질식", "기도막힘",
        "심한 흉통", "가슴을 쥐어짜는", "호흡곤란 심한", "청색증",
        "아나필락시스", "전신 두드러기 호흡곤란", "고열 경련", "간질발작",
    ],
    "응급실 방문 권고": [
        "고열", "38.5도 이상", "39도", "40도", "심한 두통", "갑작스러운 두통",
        "심한 복통", "복통 지속", "혈뇨", "혈변", "토혈 소량",
        "골절 의심", "뼈 부러짐", "심한 타박상", "관절 탈구",
        "화상 넓은", "눈 화학물질", "독극물 섭취",
        "어지러움 심한", "구토 지속", "설사 지속", "탈수",
        "흉통", "심계항진", "두근거림 심한",
    ],
    "일반 진료 권고": [],
}


@tool
def classify_emergency(symptoms: str) -> str:
    """증상을 설명하면 응급 여부를 판단하고 적절한 대응 방법을 안내합니다."""
    matched_level: str | None = None
    matched_keywords: list[str] = []

    for level, keywords in _EMERGENCY_KEYWORDS.items():
        for kw in keywords:
            if kw in symptoms:
                if matched_level is None:
                    matched_level = level
                matched_keywords.append(kw)

    if matched_level is None or matched_level == "일반 진료 권고":
        return (
            "■ 응급 분류 결과: 일반 진료 권고\n\n"
            "현재 증상은 즉각적인 응급처치가 필요한 상태로 판단되지 않습니다.\n"
            "가까운 의원이나 병원에서 진료를 받으시길 권장합니다.\n\n"
            "⚠️ 증상이 갑자기 악화되거나 호흡곤란, 의식저하, 심한 흉통이 발생하면 즉시 119에 신고하세요."
        )

    lines = [f"■ 응급 분류 결과: {matched_level}\n"]

    if matched_level == "즉시 119 신고 (생명위협)":
        lines.append("🚨 지금 즉시 119에 전화하세요!")
        lines.append(f"감지된 증상 키워드: {', '.join(matched_keywords)}\n")
        lines.append("■ 즉시 취해야 할 조치:")
        lines.append("1. 119에 신고하고 환자 위치, 증상을 정확히 전달하세요.")
        lines.append("2. 환자를 안전한 곳에 눕히고 기도를 확보하세요.")
        lines.append("3. 심정지라면 즉시 CPR(심폐소생술)을 시작하세요.")
        lines.append("4. AED(자동심장충격기)가 근처에 있다면 사용하세요.")
        lines.append("5. 구급대가 도착할 때까지 환자 곁을 지키세요.")
    else:
        lines.append("🏥 가능한 빨리 응급실을 방문하세요.")
        lines.append(f"감지된 증상 키워드: {', '.join(matched_keywords)}\n")
        lines.append("■ 응급실 방문 전 주의사항:")
        lines.append("1. 혼자 운전하지 말고 보호자와 함께 이동하세요.")
        lines.append("2. 복용 중인 약물 목록을 지참하세요.")
        lines.append("3. 증상 발생 시각과 경과를 기록해 두세요.")
        lines.append("4. 증상이 급격히 악화되면 즉시 119에 신고하세요.")

    lines.append("\n⚠️ 이 정보는 참고용이며 전문 의료진의 판단을 대체하지 않습니다.")
    return "\n".join(lines)


def _fetch_drug_interaction_text(drug_name: str) -> str | None:
    """e약은요 API에서 약물의 상호작용(intrcQesitm) 필드를 조회합니다."""
    try:
        resp = httpx.get(
            _DRUG_API_URL,
            params={
                "serviceKey": _DRUG_API_KEY,
                "itemName": drug_name,
                "type": "json",
                "numOfRows": 1,
                "pageNo": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    items: list[Any] = (
        data.get("body", {}).get("items", [])
        or data.get("response", {}).get("body", {}).get("items", [])
        or []
    )
    if isinstance(items, dict):
        items = [items]
    if not items:
        return None

    return items[0].get("intrcQesitm") or None


@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """두 약물 이름을 입력하면 식품의약품안전처 데이터를 기반으로 병용 시 상호작용·주의사항을 조회합니다."""
    intr1 = _fetch_drug_interaction_text(drug1)
    intr2 = _fetch_drug_interaction_text(drug2)

    if intr1 is None and intr2 is None:
        return (
            f"'{drug1}' 및 '{drug2}'에 대한 상호작용 정보를 찾을 수 없습니다.\n"
            "약물명이 정확한지 확인하거나 약사·의사에게 직접 문의하세요."
        )

    lines: list[str] = [f"■ '{drug1}' + '{drug2}' 병용 상호작용 정보\n"]

    if intr1:
        lines.append(f"[{drug1}의 상호작용 주의 약물]")
        lines.append(intr1)
        # drug2가 intr1 내용에 언급되는지 확인
        if drug2 in intr1:
            lines.append(f"\n⚠️ '{drug2}'이(가) '{drug1}'의 상호작용 주의 목록에 포함되어 있습니다!")
        lines.append("")

    if intr2:
        lines.append(f"[{drug2}의 상호작용 주의 약물]")
        lines.append(intr2)
        if drug1 in intr2:
            lines.append(f"\n⚠️ '{drug1}'이(가) '{drug2}'의 상호작용 주의 목록에 포함되어 있습니다!")
        lines.append("")

    lines.append("※ 병용 복용 전 반드시 의사 또는 약사와 상의하세요.")
    return "\n".join(lines).strip()


_FIRST_AID_GUIDES: dict[str, str] = {
    "화상": (
        "■ 화상 응급처치\n"
        "1. 즉시 흐르는 찬물로 15~20분 이상 식히세요 (얼음 직접 사용 금지).\n"
        "2. 물집은 터뜨리지 마세요.\n"
        "3. 화상 부위를 깨끗한 거즈나 천으로 가볍게 덮으세요.\n"
        "4. 로션, 된장, 치약 등 민간요법 사용 금지.\n"
        "5. 얼굴·손·발·생식기 화상, 넓은 범위 화상은 즉시 응급실을 방문하세요."
    ),
    "골절": (
        "■ 골절 응급처치\n"
        "1. 부러진 부위를 움직이지 마세요.\n"
        "2. 부목(나무판, 두꺼운 잡지 등)으로 골절 부위 위아래 관절을 고정하세요.\n"
        "3. 개방성 골절(뼈가 피부 밖으로 나온 경우): 깨끗한 천으로 덮고 뼈를 밀어 넣지 마세요.\n"
        "4. 출혈이 있으면 깨끗한 천으로 압박하여 지혈하세요.\n"
        "5. 즉시 119에 신고하거나 응급실을 방문하세요."
    ),
    "출혈": (
        "■ 출혈 응급처치\n"
        "1. 깨끗한 천이나 거즈로 상처를 직접 압박하세요 (최소 10분 이상).\n"
        "2. 압박 중 거즈가 흠뻑 젖으면 위에 덧대고 계속 압박하세요 (제거 금지).\n"
        "3. 가능하면 출혈 부위를 심장보다 높게 올리세요.\n"
        "4. 지혈이 안 되거나 대량출혈이면 즉시 119에 신고하세요.\n"
        "5. 지혈대는 팔다리 대량출혈 시 최후 수단으로 사용하세요."
    ),
    "심폐소생술": (
        "■ CPR(심폐소생술) 방법\n"
        "1. 환자 반응 확인: 어깨를 두드리며 '괜찮으세요?' 묻기.\n"
        "2. 119 신고 및 AED 요청.\n"
        "3. 가슴압박: 젖꼭지 연결선 중앙, 깍지 낀 두 손으로 5~6cm 깊이로 분당 100~120회 압박.\n"
        "4. 인공호흡(가능한 경우): 30회 압박 후 2회 호흡 (머리 젖히고 턱 들어 기도 확보).\n"
        "5. AED 도착 시 즉시 사용하고 구급대 도착까지 CPR 계속."
    ),
    "질식": (
        "■ 질식(기도폐쇄) 응급처치\n"
        "성인/어린이:\n"
        "1. 환자 뒤에서 허리를 감싸 안으세요.\n"
        "2. 한 손을 주먹 쥐어 배꼽과 흉골 사이에 대고 다른 손으로 감싸세요.\n"
        "3. 복부를 강하게 위쪽으로 밀어올리는 동작을 반복 (하임리히법).\n"
        "영아(1세 미만):\n"
        "1. 엎드린 자세로 등을 5회 두드리기.\n"
        "2. 뒤집어 흉부를 2손가락으로 5회 압박.\n"
        "3. 이물질이 보이면 제거, 의식 없으면 CPR 시작."
    ),
    "독극물": (
        "■ 독극물 섭취 응급처치\n"
        "1. 즉시 119 또는 중독정보센터(1588-5555)에 신고하세요.\n"
        "2. 임의로 구토를 유도하지 마세요 (산/알칼리 물질은 역효과).\n"
        "3. 섭취한 물질의 용기를 보관해 의료진에게 보여주세요.\n"
        "4. 의식이 있으면 소량의 물을 마시게 할 수 있습니다.\n"
        "5. 의식이 없거나 경련이 있으면 옆으로 눕혀 기도를 확보하세요."
    ),
    "저체온증": (
        "■ 저체온증 응급처치\n"
        "1. 환자를 따뜻하고 바람이 없는 곳으로 이동하세요.\n"
        "2. 젖은 옷을 벗기고 담요나 따뜻한 옷으로 감싸세요.\n"
        "3. 따뜻한 음료(알코올 제외)를 의식이 있을 때만 조금씩 마시게 하세요.\n"
        "4. 팔다리를 문지르지 마세요 (혈관 손상 위험).\n"
        "5. 심한 경우(떨림 없음, 의식저하) 즉시 119에 신고하세요."
    ),
    "열사병": (
        "■ 열사병 응급처치\n"
        "1. 즉시 119에 신고하세요.\n"
        "2. 서늘하고 그늘진 곳으로 이동하세요.\n"
        "3. 옷을 풀고 차가운 물수건이나 얼음팩을 목·겨드랑이·사타구니에 대세요.\n"
        "4. 의식이 있으면 시원한 물이나 이온음료를 마시게 하세요.\n"
        "5. 의식이 없으면 음료 금지, CPR 준비 후 구급대 기다리세요."
    ),
    "익사": (
        "■ 익사(물에 빠짐) 응급처치\n"
        "1. 즉시 119에 신고하세요.\n"
        "2. 구조자가 직접 뛰어들지 말고, 튜브·밧줄·막대 등을 이용해 구조하세요.\n"
        "3. 물 밖으로 꺼낸 후 의식·호흡을 확인하세요.\n"
        "4. 호흡이 없으면 즉시 CPR(심폐소생술)을 시작하세요.\n"
        "5. 물을 빼내려고 거꾸로 들거나 배를 누르지 마세요 (위 내용물 역류 위험).\n"
        "6. 의식이 있어도 저체온증 위험이 있으니 따뜻하게 하고 병원 방문 필수."
    ),
    "뇌졸중": (
        "■ 뇌졸중 응급처치 (FAST 확인법)\n"
        "F (Face): 얼굴 한쪽이 처지거나 마비되는지 확인하세요.\n"
        "A (Arms): 양팔을 들었을 때 한쪽이 처지는지 확인하세요.\n"
        "S (Speech): 말이 어눌하거나 이해하지 못하는지 확인하세요.\n"
        "T (Time): 위 증상이 하나라도 있으면 즉시 119에 신고하세요!\n\n"
        "1. 즉시 119에 신고하고 발생 시각을 기록하세요 (치료 골든타임: 4.5시간).\n"
        "2. 환자를 눕히고 머리를 약간 올려두세요.\n"
        "3. 음식·물·약 복용 금지 (삼킴 장애 위험).\n"
        "4. 의식을 잃으면 옆으로 눕혀 기도를 확보하세요.\n"
        "5. 증상이 잠깐 좋아져도 반드시 응급실로 이송하세요."
    ),
    "저혈당": (
        "■ 저혈당 응급처치\n"
        "저혈당 증상: 식은땀, 떨림, 어지러움, 두근거림, 의식 흐림\n\n"
        "의식이 있는 경우:\n"
        "1. 즉시 빠르게 흡수되는 당분을 섭취하세요.\n"
        "   - 과일주스 150mL, 사탕 3~4개, 각설탕 2~3개, 꿀 1큰술\n"
        "2. 15분 후 증상이 개선되지 않으면 한 번 더 반복하세요.\n"
        "3. 증상 회복 후 빵·밥 등 복합당질을 추가 섭취하세요.\n\n"
        "의식이 없는 경우:\n"
        "1. 입으로 음식을 주지 마세요 (질식 위험).\n"
        "2. 즉시 119에 신고하세요.\n"
        "3. 글루카곤 주사가 있으면 허벅지나 상완에 투여하세요."
    ),
    "눈 이물질": (
        "■ 눈 이물질 응급처치\n"
        "일반 이물질(먼지, 속눈썹 등):\n"
        "1. 눈을 비비지 마세요 (각막 손상 위험).\n"
        "2. 깨끗한 물이나 생리식염수로 눈을 충분히 씻어내세요.\n"
        "3. 세척 후에도 이물감이 있으면 안과를 방문하세요.\n\n"
        "화학물질(락스, 세정제 등):\n"
        "1. 즉시 흐르는 물로 15~20분 이상 눈을 씻으세요 (눈꺼풀을 강제로 열어서).\n"
        "2. 콘택트렌즈 착용 시 렌즈를 즉시 제거하고 세척하세요.\n"
        "3. 세척 후 즉시 응급실을 방문하세요."
    ),
}


@tool
def get_first_aid_guide(situation: str) -> str:
    """응급 상황(화상, 골절, 출혈, 심폐소생술, 질식, 독극물, 저체온증, 열사병 등)에 대한 응급처치 방법을 안내합니다."""
    # 키워드 매칭
    for key, guide in _FIRST_AID_GUIDES.items():
        if key in situation:
            return guide

    # 부분 매칭
    aliases: dict[str, str] = {
        "cpr": "심폐소생술", "심장": "심폐소생술", "심정지": "심폐소생술",
        "불": "화상", "뜨거운": "화상", "데": "화상",
        "뼈": "골절", "부러": "골절",
        "피": "출혈", "지혈": "출혈",
        "숨막": "질식", "목막": "질식",
        "약먹": "독극물", "농약": "독극물", "삼킴": "독극물", "먹었": "독극물",
        "추위": "저체온증", "동상": "저체온증", "얼어": "저체온증",
        "더위": "열사병", "일사병": "열사병", "폭염": "열사병",
        "물에 빠": "익사", "빠졌": "익사", "溺水": "익사",
        "뇌졸": "뇌졸중", "뇌출혈": "뇌졸중", "뇌경색": "뇌졸중", "fast": "뇌졸중",
        "저혈당": "저혈당", "혈당 낮": "저혈당", "당 떨어": "저혈당",
        "눈에 이물": "눈 이물질", "눈에 뭐가": "눈 이물질", "눈 세척": "눈 이물질",
    }
    situation_lower = situation.lower()
    for alias, key in aliases.items():
        if alias in situation_lower or alias in situation:
            return _FIRST_AID_GUIDES[key]

    available = "、".join(_FIRST_AID_GUIDES.keys())
    return (
        f"'{situation}'에 대한 응급처치 가이드를 찾지 못했습니다.\n\n"
        f"현재 안내 가능한 상황: {available}\n\n"
        "⚠️ 긴급 상황에서는 즉시 119에 신고하세요."
    )


def _is_korean_text(text: str) -> bool:
    """문자열에 한글이 포함되어 있는지 확인합니다."""
    return any("\uAC00" <= ch <= "\uD7A3" or "\u1100" <= ch <= "\u11FF" for ch in text)


@tool
def find_nearby_hospitals(location: str, specialty: str = "일반") -> str:
    """
    지역명과 병원 종별(specialty)을 기반으로 건강보험심사평가원 병원정보서비스에서 병원 목록을 조회합니다.
    location: 시도명 (예: '서울', '부산', '경기') 또는 병원명 일부
    specialty: 병원 종별 (예: '의원', '종합병원', '한의원', '치과', '일반')
    이 서비스는 대한민국 내 병원만 조회 가능합니다.
    """
    import xml.etree.ElementTree as ET

    # 한글이 없는 입력(외국 지역명 등) 조기 차단
    if not _is_korean_text(location):
        return (
            f"'{location}'은(는) 지원하지 않는 지역입니다.\n"
            "이 서비스는 대한민국 내 병원 정보만 제공합니다.\n"
            "지역명 예시: 서울, 부산, 경기, 대구, 인천, 광주, 대전, 울산, 제주"
        )

    # location → sidoCd 변환 시도 (지역명이면 코드로, 아니면 yadmNm 검색)
    sido_cd = None
    yadm_nm = None
    for key, code in _SIDO_CODE.items():
        if key in location:
            sido_cd = code
            break
    if sido_cd is None:
        # 지역명 매핑 실패 → 병원명 검색으로 fallback
        yadm_nm = location

    # specialty → clCd(종별) 또는 dgsbjtCd(진료과목) 판별
    cl_cd = _CL_CODE.get(specialty)
    dept_cd = _DEPT_CODE.get(specialty)

    params: dict[str, Any] = {
        "serviceKey": _HOSP_API_KEY,
        "pageNo": "1",
        "numOfRows": "5",
    }
    if sido_cd:
        params["sidoCd"] = sido_cd
    if yadm_nm:
        params["yadmNm"] = yadm_nm
    if dept_cd:
        params["dgsbjtCd"] = dept_cd
    elif cl_cd:
        params["clCd"] = cl_cd

    # 타임아웃 재시도 1회
    last_error: str = ""
    for attempt in range(2):
        try:
            resp = httpx.get(_HOSP_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            last_error = ""
            break
        except httpx.HTTPStatusError as e:
            return f"병원 정보 API 호출 실패 (HTTP {e.response.status_code}): {e.response.text[:200]}"
        except httpx.TimeoutException:
            last_error = "병원 정보 API 응답 시간 초과입니다. 잠시 후 다시 시도해 주세요."
        except httpx.RequestError as e:
            last_error = f"병원 정보 API 네트워크 오류: {e}"
            break
    if last_error:
        return last_error

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        return f"병원 정보 응답 파싱 오류: {e}"

    result_code = root.findtext(".//resultCode", "")
    if result_code != "00":
        result_msg = root.findtext(".//resultMsg", "")
        return f"병원 정보 API 오류 ({result_code}): {result_msg}"

    items = root.findall(".//item")
    total = root.findtext(".//totalCount", "0")

    if not items:
        return f"'{location}' 지역의 {specialty} 병원 정보를 찾을 수 없습니다."

    lines: list[str] = [
        f"■ '{location}' {specialty} 병원 목록 (전체 {total}건 중 상위 {len(items)}건)",
        "",
    ]
    for i, item in enumerate(items, 1):
        name    = item.findtext("yadmNm", "") 
        cl_name = item.findtext("clCdNm", "")
        addr    = item.findtext("addr", "")
        tel     = item.findtext("telno", "")
        dr_cnt  = item.findtext("drTotCnt", "")
        url     = item.findtext("hospUrl", "")
        lines.append(f"{i}. {name} [{cl_name}]")
        lines.append(f"   주소: {addr}")
        if tel:
            lines.append(f"   전화: {tel}")
        if dr_cnt and dr_cnt != "0":
            lines.append(f"   의사 수: {dr_cnt}명")
        if url:
            lines.append(f"   홈페이지: {url}")
        lines.append("")

    return "\n".join(lines).strip()
