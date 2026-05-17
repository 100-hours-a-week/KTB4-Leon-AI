# create_study_note.py 설명

## 개요
`create_study_note.py`는 주차 기반 폴더 구조와 날짜별 학습 노트를 자동으로 생성하는 Python 스크립트입니다. 기본 시간대는 KST(`Asia/Seoul`)이며, 부트캠프 시작일을 기준으로 주차를 계산해 `weeks/week-XX` 디렉터리를 만듭니다.

## 지원 노트 타입
- `til` : Daily TIL(오늘 배운 것) 템플릿
- `deep-dive` : 금요일 딥다이브 세션 정리 템플릿
- `weekly-challenge` : 토요일 주간 챌린지 기록 템플릿

## 주요 동작 원리(함수별 요약)
- `parse_args()` : 커맨드라인 인자 파싱
- `get_target_date(value)` : `--date`가 주어지면 해당 날짜, 없으면 현재 KST 날짜 반환
- `get_week_number(target_date, start_date)` : 시작일 기준 주차(1부터) 계산
- `format_date(target_date)` : YYYY-M-D 형식 문자열 생성
- `ensure_week_readme(week_dir, week_number)` : 주차 README 생성(템플릿 존재 시 사용)
- `ensure_section_readme(section_dir, title, description)` : 섹션 README 생성
- `build_*_content(target_date)` : 각 노트 타입별 마크다운 내용 생성
- `create_note(note_type, target_date, start_date)` : 디렉터리 생성, README 보장, 노트 파일 작성

## 사용법
프로젝트 루트에서 실행하세요:

```bash
python 01/create_study_note.py --type til
python 01/create_study_note.py --type deep-dive --date 2026-05-14
python 01/create_study_note.py --type weekly-challenge --start-date 2026-05-14
```

옵션 설명:
- `--type` : 필수. `til`, `deep-dive`, `weekly-challenge` 중 선택
- `--date` : 선택. 타깃 날짜(YYYY-MM-DD), 미지정 시 현재 KST 날짜 사용
- `--start-date` : 선택. 부트캠프 시작일(YYYY-MM-DD). 기본값은 환경변수 `BOOTCAMP_START_DATE` 또는 코드 내 `DEFAULT_START_DATE` (`2026-05-14`)

## 출력 예시
- `weeks/week-01/README.md`
- `weeks/week-01/daily/README.md`
- `weeks/week-01/daily/TIL 2026-05-17.md`

## 템플릿 파일
- `templates/weekly-review.md`가 있으면 주차 README 생성 시 첫 번째 `# Week XX` 헤더를 치환해 사용합니다.

## 요구 사항
- Python 3.9 이상 권장(표준 라이브러리의 `zoneinfo` 사용)

## 수정 포인트
- 노트 내용 포맷을 바꾸려면 스크립트 내부의 `build_til_content`, `build_deep_dive_content`, `build_weekly_challenge_content` 함수를 수정하세요.

---
작성자: 자동 생성된 설명서 — 추가 수정을 원하시면 알려주세요.
