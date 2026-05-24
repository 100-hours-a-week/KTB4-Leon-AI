# Week 2 Community Service

초보자가 구조를 확인하기 쉽도록 최대한 단순하게 만든 FastAPI + SQLite 예제입니다.

## 실행 방법

```bash
cd weeks/week-02/weekly-challenge/community-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

브라우저에서 아래 주소를 확인하세요.

- API 문서: http://127.0.0.1:8000/docs
- 간단 프론트엔드: `frontend/index.html` 파일을 브라우저로 열기

## 데이터베이스는 어디에 생기나요?

서버를 실행하면 `community.db` 파일이 자동으로 생깁니다.

`app/database.py`에서 SQLite에 연결하고, `posts` 테이블을 만듭니다.

## 처음 볼 파일 순서

처음에는 아래 순서대로 코드를 따라가면 됩니다.

1. `app/main.py`: FastAPI 앱이 어디서 시작되는지 확인합니다.
2. `app/schemas.py`: 게시글 데이터가 어떤 모양인지 확인합니다.
3. `app/database.py`: SQLite DB 연결과 테이블 생성 코드를 확인합니다.
4. `app/routes/posts.py`: `/posts` 주소가 어떤 함수로 연결되는지 확인합니다.
5. `app/controllers/posts_controller.py`: 실제 DB 조회, 저장, 삭제 로직을 확인합니다.
6. `frontend/index.html`: 브라우저 화면에서 API를 어떻게 호출하는지 확인합니다.

처음부터 모든 파일을 외우려고 하지 말고, `main.py`에서 시작해서 `routes`, `controllers`, `database` 순서로 따라가면 흐름이 보입니다.

## 구조

```text
app/
  main.py
  database.py
  schemas.py
  routes/
    posts.py
  controllers/
    posts_controller.py
frontend/
  index.html
```

역할은 아래처럼 보면 됩니다.

- `main.py`: FastAPI 앱 시작점
- `routes/posts.py`: URL 주소를 받는 곳
- `controllers/posts_controller.py`: 실제 기능을 처리하는 곳
- `database.py`: DB 연결, 테이블 생성, SQL 실행
- `schemas.py`: 요청/응답 데이터 모양 정의
