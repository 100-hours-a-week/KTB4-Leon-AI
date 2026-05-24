# Weekly Challenge

Week 02 위클리 챌린지 과제를 저장하는 공간입니다.

---

## 과제 목록

| 과제 | 파일 | 설명 |
| --- | --- | --- |
| HTTP 내용 정리 | [HTTP.md](./HTTP.md) | HTTP 개념, 요청/응답 구조, 메서드, 상태 코드 정리 |
| FastAPI 커뮤니티 백엔드 구현 | [community-service/](./community-service/) | Router, Controller, Model 구조로 분리한 커뮤니티 서비스 구현 |
| 프론트엔드 예제 | [community-service/frontend/](./community-service/frontend/) | 게시글 작성, 목록, 상세 조회를 테스트하는 간단한 화면 |

---

2.4 파일 여러개로 쪼개기 구조개선하기
Router - Controller - Model


2.5 (선택) HTMLICSSJJS나 스트림릿 으로 프론트엔드 만들기
글 쓰기 
글 목록
글 상세: 댓글 목록
ent to ent
user ent - frontend - ai backend - db llm end
bridge builder
interface design

## 과제 내용

## 실행 방법

```bash
cd /Users/samrobert/Documents/GitHub/KakaoBootCamp
python -m venv .venv
source .venv/bin/activate
python -m pip install -r weeks/week-02/weekly-challenge/community-service/requirements.txt
cd weeks/week-02/weekly-challenge/community-service
uvicorn app.main:app --reload
```

서버가 실행되면 `http://127.0.0.1:8000/docs`에서 API를 테스트할 수 있습니다.

### 1. HTTP 내용 정리

HTTP와 관련된 핵심 개념을 정리합니다.

정리할 내용 예시:

- HTTP란 무엇인가
- HTTP 요청과 응답 구조
- HTTP 메서드
- HTTP 상태 코드
- REST API 기본 개념

---

### 2. FastAPI로 커뮤니티 서비스의 백엔드 구현

FastAPI를 사용해 커뮤니티 서비스의 백엔드를 구현합니다.

#### 구현 목표

- HTTP REST API 설계
- FastAPI 기반 API 구현
- 게시글 생성, 조회, 수정, 삭제 기능 구현
- 필요한 경우 댓글 또는 사용자 관련 기능 추가
