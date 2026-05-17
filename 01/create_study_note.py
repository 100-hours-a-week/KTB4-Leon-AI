import json
from pathlib import Path

DATA_FILE = Path(__file__).with_name("study_records.json")

def load_records():
    if not DATA_FILE.exists():
        return [], []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    todos = data.get("todos", [])
    learnings = data.get("learnings", [])
    return todos, learnings


def save_records(todos, learnings):
    data = {
        "todos": todos,
        "learnings": learnings,
    }

    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def print_menu():
    print()
    print("=== Kakao Boot Camp Study CLI ===")
    print("1. 오늘 할 일 추가")
    print("2. 오늘 배운 내용 추가")
    print("3. 현재 기록 보기")
    print("4. 종료")


def print_records(todos, learnings):
    print()
    print("----- 오늘의 기록 -----")

    print("[오늘 할 일]")
    if todos:
        for index, todo in enumerate(todos, start=1):
            print(f"{index}. {todo}")
    else:
        print("아직 등록된 할 일이 없습니다.")

    print()
    print("[오늘 배운 내용]")
    if learnings:
        for index, learning in enumerate(learnings, start=1):
            print(f"{index}. {learning}")
    else:
        print("아직 등록된 학습 내용이 없습니다.")


def main():
    todos, learnings = load_records()

    print("카카오 부트캠프 학습 기록 CLI입니다.")

    while True:
        print_menu()
        choice = input("메뉴를 선택하세요: ").strip()

        if choice == "1":
            todo = input("오늘 할 일을 입력하세요: ").strip()
            if todo:
                todos.append(todo)
                save_records(todos, learnings)
                print("할 일이 추가되었습니다.")
            else:
                print("빈 내용은 추가할 수 없습니다.")
        elif choice == "2":
            learning = input("오늘 배운 내용을 입력하세요: ").strip()
            if learning:
                learnings.append(learning)
                save_records(todos, learnings)
                print("학습 내용이 추가되었습니다.")
            else:
                print("빈 내용은 추가할 수 없습니다.")
        elif choice == "3":
            print_records(todos, learnings)
        elif choice == "4":
            save_records(todos, learnings)
            print("오늘도 수고했습니다. 기록을 저장하고 마칩니다.")
            break
        else:
            print("1부터 4까지의 숫자 중에서 선택해 주세요.")


if __name__ == "__main__":
    main()