from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
DEFAULT_START_DATE = "2026-05-14"
NOTE_TYPES = ("til", "deep-dive", "weekly-challenge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dated study notes.")
    parser.add_argument(
        "--type",
        choices=NOTE_TYPES,
        required=True,
        help="Type of study note to create.",
    )
    parser.add_argument(
        "--date",
        help="Target date in YYYY-MM-DD format. Defaults to today's KST date.",
    )
    parser.add_argument(
        "--start-date",
        default=os.getenv("BOOTCAMP_START_DATE", DEFAULT_START_DATE),
        help="Boot camp start date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def get_target_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now(KST).date()


def get_week_number(target_date: date, start_date: date) -> int:
    start_monday = start_date - timedelta(days=start_date.weekday())
    target_monday = target_date - timedelta(days=target_date.weekday())
    return max(((target_monday - start_monday).days // 7) + 1, 1)


def format_date(target_date: date) -> str:
    return f"{target_date.year}-{target_date.month}-{target_date.day}"


def ensure_week_readme(week_dir: Path, week_number: int) -> None:
    readme = week_dir / "README.md"
    if readme.exists():
        return

    template = Path("templates/weekly-review.md")
    if template.exists():
        content = template.read_text(encoding="utf-8")
        content = content.replace("# Week XX", f"# Week {week_number:02d}", 1)
    else:
        content = f"# Week {week_number:02d}\n\n## 이번 주 목표\n\n- [ ] \n"

    readme.write_text(content, encoding="utf-8")


def ensure_section_readme(section_dir: Path, title: str, description: str) -> None:
    readme = section_dir / "README.md"
    if readme.exists():
        return

    readme.write_text(f"# {title}\n\n{description}\n", encoding="utf-8")


def build_til_content(target_date: date) -> str:
    label = format_date(target_date)
    return f"""# TIL {label}

## 오늘 할 일

- [ ]
- [ ]
- [ ]

## 오늘 배운 내용

-

## 헷갈렸던 내용

-

## 해결한 문제

-

## 내일 할 일

- [ ]
- [ ]
"""


def build_deep_dive_content(target_date: date) -> str:
    label = format_date(target_date)
    return f"""# Deep Dive {label}

## 세션 정보

- 시간: 금요일 15:00-15:50
- 장소: 온라인 Zoom 또는 Discord
- 방식: 6개의 주제 중 1-2개를 골라 팀별 심층 연구 후 발표

## 선택 주제

- [ ]
- [ ]

## 조사 내용

-

## 발표 정리

-

## 참고 자료

- Notion 강의 자료:
- 세션 기록:

## 다음에 더 볼 것

-
"""


def build_weekly_challenge_content(target_date: date) -> str:
    label = format_date(target_date)
    return f"""# Weekly Challenge {label}

## 일정

- 시작: 토요일 오후 17:00-20:00 사이
- 제출 마감: 일요일 23:59
- 장소: 온라인 Notion, Discord 또는 가상 회의

## 과제

- CLI 프로그램 제작
- Python `input()`과 `print()`를 사용한 콘솔 프로그램
- 필요하면 `argparse` 또는 `click` 활용 가능

## 진행 체크리스트

- [ ] 문제 요구사항 읽기
- [ ] 기능 목록 작성하기
- [ ] 기본 입출력 구현하기
- [ ] 예외 상황 점검하기
- [ ] 제출 전 실행 확인하기

## 구현 메모

-

## 제출 링크

-
"""


def create_note(note_type: str, target_date: date, start_date: date) -> Path:
    week_number = get_week_number(target_date, start_date)
    week_dir = Path("weeks") / f"week-{week_number:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)
    ensure_week_readme(week_dir, week_number)

    date_label = format_date(target_date)

    if note_type == "til":
        section_dir = week_dir / "daily"
        filename = f"TIL {date_label}.md"
        content = build_til_content(target_date)
        section_dir.mkdir(parents=True, exist_ok=True)
        ensure_section_readme(
            section_dir,
            "Daily Logs",
            "날짜별 TIL과 학습 기록을 저장하는 공간입니다.",
        )
    elif note_type == "deep-dive":
        section_dir = week_dir / "deep-dive"
        filename = f"Deep Dive {date_label}.md"
        content = build_deep_dive_content(target_date)
        section_dir.mkdir(parents=True, exist_ok=True)
        ensure_section_readme(
            section_dir,
            "Deep Dive",
            "금요일 딥다이브 세션의 주제, 조사 내용, 발표 기록을 저장하는 공간입니다.",
        )
    else:
        section_dir = week_dir / "weekly-challenge"
        filename = f"Weekly Challenge {date_label}.md"
        content = build_weekly_challenge_content(target_date)
        section_dir.mkdir(parents=True, exist_ok=True)
        ensure_section_readme(
            section_dir,
            "Weekly Challenge",
            "토요일 위클리 챌린지 진행 내용과 제출 기록을 저장하는 공간입니다.",
        )

    note_path = section_dir / filename
    if not note_path.exists():
        note_path.write_text(content, encoding="utf-8")

    return note_path


def main() -> None:
    args = parse_args()
    target_date = get_target_date(args.date)
    start_date = date.fromisoformat(args.start_date)
    note_path = create_note(args.type, target_date, start_date)
    print(f"Created or already exists: {note_path}")


if __name__ == "__main__":
    main()