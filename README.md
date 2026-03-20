# Medical AI Assistant

FastAPI + LangGraph 기반의 의료 AI 어시스턴트입니다.

## 기술 스택

- FastAPI
- LangGraph / LangChain
- OpenAI (GPT-4o)
- Elasticsearch (증상 검색 RAG)
- uv (패키지 관리)

## 환경 준비 및 설치 가이드

### 1. 사전 요구사항

Python 3.11 이상 3.13 이하 버전을 권장합니다.

```bash
# uv 패키지 매니저 설치 (macOS / Linux / Windows WSL)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 의존성 설치

```bash
uv sync
```

### 3. 환경 변수 설정

`env.sample`을 복사하여 `.env`를 생성하고 필요한 값을 입력합니다.

```bash
cp env.sample .env
```

| 변수 | 필수 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API 키 |
| `OPENAI_MODEL` | Yes | 예: `gpt-4o` |
| `API_V1_PREFIX` | Yes | 예: `/api/v1` |
| `ES_URL` | No | Elasticsearch URL (기본값: `https://elasticsearch-edu.didim365.app`) |
| `ES_USER` | No | Elasticsearch 사용자명 (기본값: `elastic`) |
| `ES_PASSWORD` | No | Elasticsearch 비밀번호 |
| `ES_INDEX_NAME` | No | Elasticsearch 인덱스 (기본값: `edu-collection`) |
| `MFDS_API_KEY` | No | 식품의약품안전처 공공데이터 API 키 (e약은요) |
| `HIRA_API_KEY` | No | 건강보험심사평가원 API 키 (병원정보) |
| `DEEPAGENT_RECURSION_LIMIT` | No | LangGraph 최대 재귀 횟수 (기본값: 20) |
| `OPIK__URL_OVERRIDE` | No | Opik 서버 URL (자체 호스팅) |
| `OPIK__API_KEY` | No | Opik Cloud API 키 |
| `OPIK__WORKSPACE` | No | Opik 워크스페이스 이름 |
| `OPIK__PROJECT` | No | Opik 프로젝트 이름 |

### 4. 개발 서버 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

서버 구동 후 `http://localhost:8000/docs` 에서 API 문서를 확인할 수 있습니다.

## API 엔드포인트

- `GET /` — API 정보
- `GET /health` — 헬스 체크
- `GET /api/v1/threads` — 대화 스레드 목록 조회
- `GET /api/v1/threads/{thread_id}` — 스레드 상세 및 메시지 조회
- `POST /api/v1/chat` — 메시지 전송 (SSE 스트리밍)

`POST /api/v1/chat` 응답은 SSE(`text/event-stream`) 형식이며, 각 이벤트는 `step` 필드를 포함합니다:
- `{"step": "model", "tool_calls": [...]}` — 도구 호출 예정
- `{"step": "tools", "name": "...", "content": {...}}` — 도구 실행 결과
- `{"step": "done", "message_id": "...", "role": "assistant", "content": "...", "metadata": {...}}` — 최종 답변

## 에이전트 도구

| 도구 | 설명 |
|---|---|
| `search_symptoms` | 증상 기반 Elasticsearch RAG 검색 + LLM 요약 |
| `get_medication_info` | 식품의약품안전처 e약은요 약물 정보 조회 (영어 약이름 지원) |
| `find_nearby_hospitals` | 건강보험심사평가원 지역·진료과목별 병원 검색 |
| `check_drug_interaction` | 두 약물의 상호작용 및 병용 주의사항 조회 |
| `classify_emergency` | 자연어 증상 기반 응급도 LLM 판단 (즉시119/응급실/일반진료) |
| `get_first_aid_guide` | 응급 상황별 응급처치 방법 안내 |

## 주요 명령어

```bash
# 테스트 실행
uv run pytest

# 단일 테스트 실행
uv run pytest tests/test_main.py::test_health -v

# Opik 평가 데이터셋 생성
uv run python -m app.eval.create_dataset --dataset yja-dataset

# Opik 평가 실행 (데이터셋 생성 후)
uv run python -m app.eval.run_opik_eval --dataset yja-dataset --experiment exp-v1

# 린트 / 포맷
uv run ruff check .
uv run black .
```
