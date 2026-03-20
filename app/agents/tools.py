from __future__ import annotations

from typing import Any

import httpx
from elasticsearch import Elasticsearch
from langchain.tools import tool
from langchain_elasticsearch import ElasticsearchRetriever

from app.core.config import settings

# ---------------------------------------------------------------------------
# Elasticsearch 연결 설정
# ---------------------------------------------------------------------------

_CONTENT_FIELD = "content"
_TOP_K = 5

# 식품의약품안전처(e약은요) API
_DRUG_API_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"

# 식품의약품안전처 DUR 병용금기 API
_DUR_API_URL = "https://apis.data.go.kr/1471000/DURInfrService01/getUsjntTabooInfoList01"

# 건강보험심사평가원 병원정보서비스 API
_HOSP_API_URL = "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"

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
        settings.ES_URL,
        basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
        verify_certs=False,
    )
    return ElasticsearchRetriever(
        index_name=settings.ES_INDEX_NAME,
        body_func=_bm25_query,
        content_field=_CONTENT_FIELD,
        client=es_client,
    )


# 전역 싱글턴 인스턴스
_retriever: ElasticsearchRetriever | None = None


def _get_retriever() -> ElasticsearchRetriever:
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever()
    else:
        # 연결 상태 확인 후 실패 시 재초기화
        try:
            _retriever.client.info()
        except Exception:
            _retriever = _build_retriever()
    return _retriever


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_symptoms(symptoms: str) -> str:
    """주어진 증상을 기반으로 의료 지식 베이스에서 관련 정보를 검색하고, 증상과 관련된 핵심 내용을 요약해 반환합니다."""
    retriever = _get_retriever()
    docs = retriever.invoke(symptoms)
    if not docs:
        return f"증상 '{symptoms}'에 대한 관련 의료 정보를 찾을 수 없습니다."

    raw_chunks: list[str] = []
    for i, doc in enumerate(docs, 1):
        source_spec = doc.metadata.get("_source", {}).get("source_spec", "unknown")
        creation_year = doc.metadata.get("_source", {}).get("creation_year", "")
        header = f"[{i}] 출처: {source_spec}" + (f" ({creation_year}년)" if creation_year and creation_year != "null" else "")
        snippet = doc.page_content[:600].replace("\n", " ")
        raw_chunks.append(f"{header}\n{snippet}")

    raw_text = "\n\n".join(raw_chunks)

    # LLM으로 증상과 관련된 핵심 정보만 추출·요약
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()
    system = (
        "당신은 의료 정보 전문가입니다. 주어진 검색 결과에서 사용자의 증상과 직접 관련된 "
        "핵심 의료 정보만 추출해 간결하게 요약하세요. "
        "관련 없는 내용은 제외하고, 원인·특징적 증상·진단 단서·치료 방향·주의사항을 중심으로 정리하세요. "
        "출처 정보는 마지막에 간략히 명시하세요."
    )
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"증상: {symptoms}\n\n검색 결과:\n{raw_text}"),
    ])
    return response.content.strip()

# 영어 약이름 → 한국어 검색어 매핑 (e약은요 API는 한국어 제품명 기반)
_DRUG_NAME_ALIASES: dict[str, str] = {
    # 진통·해열제
    "aspirin": "아스피린",
    "acetaminophen": "아세트아미노펜",
    "paracetamol": "아세트아미노펜",
    "tylenol": "타이레놀",
    "ibuprofen": "이부프로펜",
    "advil": "이부프로펜",
    "motrin": "이부프로펜",
    "naproxen": "나프록센",
    "diclofenac": "디클로페낙",
    "celecoxib": "셀레콕시브",
    "tramadol": "트라마돌",
    # 항생제
    "amoxicillin": "아목시실린",
    "augmentin": "아목시실린",
    "azithromycin": "아지트로마이신",
    "zithromax": "아지트로마이신",
    "ciprofloxacin": "시프로플록사신",
    "doxycycline": "독시사이클린",
    "clarithromycin": "클래리스로마이신",
    "cephalexin": "세파렉신",
    "levofloxacin": "레보플록사신",
    "metronidazole": "메트로니다졸",
    # 당뇨약
    "metformin": "메트포르민",
    "glucophage": "메트포르민",
    "glipizide": "글리피자이드",
    "sitagliptin": "시타글립틴",
    "januvia": "시타글립틴",
    "empagliflozin": "엠파글리플로진",
    "jardiance": "엠파글리플로진",
    "insulin": "인슐린",
    # 심혈관
    "warfarin": "와파린",
    "coumadin": "와파린",
    "amlodipine": "암로디핀",
    "norvasc": "암로디핀",
    "atorvastatin": "아토르바스타틴",
    "lipitor": "아토르바스타틴",
    "rosuvastatin": "로수바스타틴",
    "crestor": "로수바스타틴",
    "losartan": "로사르탄",
    "cozaar": "로사르탄",
    "valsartan": "발사르탄",
    "lisinopril": "리시노프릴",
    "carvedilol": "카르베딜롤",
    "bisoprolol": "비소프롤롤",
    "clopidogrel": "클로피도그렐",
    "plavix": "클로피도그렐",
    "furosemide": "푸로세미드",
    "lasix": "푸로세미드",
    # 위장약
    "omeprazole": "오메프라졸",
    "esomeprazole": "에소메프라졸",
    "nexium": "에소메프라졸",
    "pantoprazole": "판토프라졸",
    "lansoprazole": "란소프라졸",
    "ranitidine": "라니티딘",
    "domperidone": "돔페리돈",
    "metoclopramide": "메토클로프라미드",
    # 호흡기·알레르기
    "cetirizine": "세티리진",
    "zyrtec": "세티리진",
    "loratadine": "로라타딘",
    "claritin": "로라타딘",
    "fexofenadine": "펙소페나딘",
    "allegra": "펙소페나딘",
    "montelukast": "몬테루카스트",
    "singulair": "몬테루카스트",
    "salbutamol": "살부타몰",
    "albuterol": "살부타몰",
    "fluticasone": "플루티카손",
    # 정신·신경계
    "alprazolam": "알프라졸람",
    "xanax": "알프라졸람",
    "diazepam": "디아제팜",
    "valium": "디아제팜",
    "zolpidem": "졸피뎀",
    "ambien": "졸피뎀",
    "sertraline": "설트랄린",
    "zoloft": "설트랄린",
    "fluoxetine": "플루옥세틴",
    "prozac": "플루옥세틴",
    "escitalopram": "에스시탈로프람",
    "lexapro": "에스시탈로프람",
    "pregabalin": "프레가발린",
    "lyrica": "프레가발린",
    "gabapentin": "가바펜틴",
    # 기타
    "dexamethasone": "덱사메타손",
    "prednisolone": "프레드니솔론",
    "levothyroxine": "레보티록신",
    "synthroid": "레보티록신",
    "allopurinol": "알로퓨리놀",
    "sildenafil": "실데나필",
    "viagra": "실데나필",
    "tadalafil": "타다라필",
    "cialis": "타다라필",
}


def _strip_html(text: str) -> str:
    """API 응답에서 HTML 태그를 제거합니다."""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()


def _get_llm():
    """동기 ChatOpenAI 인스턴스를 반환합니다 (도구 내부 LLM 호출용)."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )


@tool
def get_medication_info(medication_name: str) -> str:
    """약물 이름을 받아 식품의약품안전처(e약은요) API에서 효능, 사용법, 주의사항, 부작용, 보관법 등을 조회합니다."""
    # 영어 약이름이면 한국어로 변환
    search_name = _DRUG_NAME_ALIASES.get(medication_name.lower().strip(), medication_name.strip())

    try:
        resp = httpx.get(
            _DRUG_API_URL,
            params={
                "serviceKey": settings.MFDS_API_KEY,
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

@tool
def classify_emergency(symptoms: str) -> str:
    """증상을 자연어로 설명하면 응급의학 기준으로 응급 여부를 판단하고 적절한 대응 방법을 안내합니다."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = """당신은 응급의학과 전문의입니다. 환자의 증상 설명을 듣고 응급도를 판단하세요.

분류 기준:
- 즉시119: 생명에 위협적인 상황 (심정지, 호흡정지, 의식불명, 대량출혈, 뇌졸중 의심, 아나필락시스, 심한 흉통으로 쥐어짜는 느낌, 경련 등)
- 응급실: 빠른 처치가 필요한 상황 (38.5도 이상 고열, 심한 복통, 골절 의심, 독극물 섭취, 흉통, 지속되는 구토·설사·어지러움 등)
- 일반진료: 가까운 의원에서 진료 가능한 상황

다음 형식으로 정확히 답변하세요 (형식을 절대 바꾸지 마세요):
분류: [즉시119/응급실/일반진료]
판단근거: [증상에서 응급도를 판단한 구체적 이유 1~2문장]
조치:
1. [첫 번째 취해야 할 행동]
2. [두 번째 취해야 할 행동]
3. [세 번째 취해야 할 행동]"""

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"환자 증상: {symptoms}"),
    ])
    raw = response.content.strip()

    # 응답 파싱
    level = "일반진료"
    reason = ""
    action_lines: list[str] = []
    in_actions = False

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("분류:"):
            val = line.replace("분류:", "").strip()
            if "즉시119" in val or "즉시 119" in val:
                level = "즉시119"
            elif "응급실" in val:
                level = "응급실"
        elif line.startswith("판단근거:"):
            reason = line.replace("판단근거:", "").strip()
        elif line.startswith("조치:"):
            in_actions = True
        elif in_actions and line:
            action_lines.append(line)

    # 출력 포맷
    if level == "즉시119":
        out = "■ 응급 분류 결과: 즉시 119 신고 (생명위협)\n\n"
        out += "🚨 지금 즉시 119에 전화하세요!\n\n"
        if reason:
            out += f"판단 근거: {reason}\n\n"
        if action_lines:
            out += "■ 즉시 취해야 할 조치:\n"
            out += "\n".join(action_lines)
    elif level == "응급실":
        out = "■ 응급 분류 결과: 응급실 방문 권고\n\n"
        out += "가능한 빨리 응급실을 방문하세요.\n\n"
        if reason:
            out += f"판단 근거: {reason}\n\n"
        if action_lines:
            out += "■ 응급실 방문 전 주의사항:\n"
            out += "\n".join(action_lines)
    else:
        out = "■ 응급 분류 결과: 일반 진료 권고\n\n"
        out += "현재 증상은 즉각적인 응급처치가 필요한 상태로 판단되지 않습니다.\n"
        out += "가까운 의원이나 병원에서 진료를 받으시길 권장합니다.\n"
        if reason:
            out += f"\n판단 근거: {reason}\n"

    out += "\n\n※ AI 판단은 참고용입니다. 증상이 의심스러우면 즉시 119에 신고하거나 응급실을 방문하세요."
    return out


def _fetch_drug_interaction_info(drug_name: str) -> dict[str, str]:
    """e약은요 API에서 약물의 상호작용(intrcQesitm) 및 주의사항 정보를 조회합니다."""
    try:
        resp = httpx.get(
            _DRUG_API_URL,
            params={
                "serviceKey": settings.MFDS_API_KEY,
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
        return {}

    items: list[Any] = (
        data.get("body", {}).get("items", [])
        or data.get("response", {}).get("body", {}).get("items", [])
        or []
    )
    if isinstance(items, dict):
        items = [items]
    if not items:
        return {}

    item = items[0]
    return {
        "item_name": item.get("itemName", drug_name),
        "intrc": _strip_html(item.get("intrcQesitm") or ""),
        "atpn": _strip_html(item.get("atpnQesitm") or ""),
        "atpn_warn": _strip_html(item.get("atpnWarnQesitm") or ""),
    }


@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """두 약물 이름을 입력하면 식품의약품안전처 e약은요 데이터를 기반으로 상호작용 및 병용 주의사항을 조회합니다."""
    name1 = _DRUG_NAME_ALIASES.get(drug1.lower().strip(), drug1.strip())
    name2 = _DRUG_NAME_ALIASES.get(drug2.lower().strip(), drug2.strip())

    info1 = _fetch_drug_interaction_info(name1)
    info2 = _fetch_drug_interaction_info(name2)

    lines: list[str] = [f"■ '{drug1}' + '{drug2}' 약물 상호작용 조회\n"]
    found_cross = False

    # drug1의 상호작용 정보에서 drug2 언급 확인
    intrc1 = info1.get("intrc", "")
    if intrc1 and (name2 in intrc1 or name2.lower() in intrc1.lower()):
        found_cross = True
        lines.append(f"[{drug1} 상호작용 정보에서 {drug2} 언급 확인]")
        lines.append(intrc1)
        lines.append("")

    # drug2의 상호작용 정보에서 drug1 언급 확인
    intrc2 = info2.get("intrc", "")
    if intrc2 and (name1 in intrc2 or name1.lower() in intrc2.lower()):
        found_cross = True
        lines.append(f"[{drug2} 상호작용 정보에서 {drug1} 언급 확인]")
        lines.append(intrc2)
        lines.append("")

    if found_cross:
        lines.insert(1, "⚠️ 두 약물 간 상호작용 정보가 확인되었습니다. 반드시 의사·약사와 상의하세요.\n")
    else:
        # 직접 언급 없음 → 각 약물의 전체 상호작용 주의사항 제공
        has_info = False
        if intrc1:
            has_info = True
            lines.append(f"[{drug1}의 상호작용 주의사항]")
            lines.append(intrc1)
            lines.append("")
        if intrc2:
            has_info = True
            lines.append(f"[{drug2}의 상호작용 주의사항]")
            lines.append(intrc2)
            lines.append("")
        if has_info:
            lines.append("두 약물 간 직접적인 상호작용 언급은 없으나, 위 주의사항을 참고하시고 필요 시 의사·약사와 상담하세요.")
        else:
            lines.append(
                "두 약물의 상호작용 정보를 조회할 수 없습니다.\n"
                "약물명이 정확한지 확인하거나 약사·의사에게 직접 문의하세요."
            )

    lines.append("\n※ 이 정보는 식품의약품안전처 e약은요 데이터 기반이며, 최종 판단은 전문 의료진에게 문의하세요.")
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

    # DB 미매칭 → LLM이 응급처치 가이드 생성
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()
    system = (
        "당신은 응급처치 전문가입니다. 요청한 상황에 대한 응급처치 방법을 "
        "번호 목록으로 구체적이고 명확하게 안내하세요. "
        "마지막에는 반드시 119 신고 관련 안내를 포함하세요."
    )
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"'{situation}' 상황에서의 응급처치 방법을 안내해주세요."),
    ])
    return (
        f"■ {situation} 응급처치\n\n"
        f"{response.content.strip()}\n\n"
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
        "serviceKey": settings.HIRA_API_KEY,
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
        name     = item.findtext("yadmNm", "")
        cl_name  = item.findtext("clCdNm", "")
        addr     = item.findtext("addr", "")
        tel      = item.findtext("telno", "")
        dr_cnt   = item.findtext("drTotCnt", "")
        url      = item.findtext("hospUrl", "")
        # 응급실 여부 (emclsName: 응급의료기관 종류명)
        emcls    = item.findtext("emclsName", "")
        # 진료 요일 (dgidIdName: 진료과목 및 운영일 정보가 있는 경우)
        # HIRA API 필드: 평일/토요일/일요일/공휴일 진료 여부
        wkday    = item.findtext("wkday", "")   # 평일 운영 시간
        satday   = item.findtext("sat", "")     # 토요일 운영 시간
        sunHoli  = item.findtext("sun", "")     # 일요일 운영 시간
        holiday  = item.findtext("holi", "")    # 공휴일 운영 시간
        lunchYn  = item.findtext("lunchYn", "") # 점심시간 진료 여부 (Y/N)

        lines.append(f"{i}. {name} [{cl_name}]" + (f" [응급의료기관: {emcls}]" if emcls else ""))
        lines.append(f"   주소: {addr}")
        if tel:
            lines.append(f"   전화: {tel}")
        if dr_cnt and dr_cnt != "0":
            lines.append(f"   의사 수: {dr_cnt}명")

        # 진료시간 정리
        hours_info: list[str] = []
        if wkday:
            hours_info.append(f"평일 {wkday}")
        if satday:
            hours_info.append(f"토 {satday}")
        if sunHoli:
            hours_info.append(f"일 {sunHoli}")
        if holiday:
            hours_info.append(f"공휴일 {holiday}")
        if hours_info:
            lines.append(f"   진료시간: {' / '.join(hours_info)}")
        if lunchYn == "Y":
            lines.append("   점심시간 진료: 가능")

        if url:
            lines.append(f"   홈페이지: {url}")
        lines.append("")

    return "\n".join(lines).strip()
