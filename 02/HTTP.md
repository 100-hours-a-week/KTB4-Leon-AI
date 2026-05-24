# HTTP REST API 설계

## 서비스 주제

FastAPI 기반 커뮤니티 서비스 백엔드입니다.

사용자는 게시글을 작성할 수 있습니다.
게시글은 REST 방식으로 생성, 조회, 삭제할 수 있습니다.

## 실행 방법

```bash
cd /Users/samrobert/Documents/GitHub/KakaoBootCamp
python -m venv .venv
source .venv/bin/activate
python -m pip install -r weeks/week-02/weekly-challenge/community-service/requirements.txt
cd weeks/week-02/weekly-challenge/community-service
uvicorn app.main:app --reload
```

서버 실행 후 브라우저에서 `http://127.0.0.1:8000/docs`를 열면 Swagger 문서로 API를 테스트할 수 있습니다.

## 리소스

| 리소스 | 설명 |
|---|---|
| Post | 커뮤니티 게시글 |

## API 목록

| Method | Endpoint | 설명 |
|---|---|---|
| GET | `/` | API 상태 확인 |
| POST | `/posts` | 게시글 생성 |
| GET | `/posts` | 게시글 목록 조회 |
| GET | `/posts/{post_id}` | 게시글 단건 조회 |
| DELETE | `/posts/{post_id}` | 게시글 삭제 |
| GET | `/ai/recommend-title` | 키워드 기반 제목 추천 |

## 상태 코드

| Status Code | 의미 | 사용 예시 |
|---|---|---|
| 200 | 요청 성공 | 조회, 삭제 성공 |
| 201 | 생성 성공 | 게시글 생성 |
| 404 | 리소스 없음 | 없는 게시글 조회 |
| 422 | 검증 실패 | 필수 값 누락, 빈 문자열 입력 |

## 요청 예시

```bash
curl -X POST http://127.0.0.1:8000/posts \
  -H "Content-Type: application/json" \
  -d '{"title": "HTTP REST API", "content": "FastAPI로 커뮤니티 API를 구현합니다."}'
```

```bash
curl http://127.0.0.1:8000/posts/1
```

```bash
curl "http://127.0.0.1:8000/ai/recommend-title?keyword=FastAPI"
```
