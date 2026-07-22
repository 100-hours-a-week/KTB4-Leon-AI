# 9주차 위클리 챌린지: Qwen Fine-Tuning과 양자화

## Colab 실행 파일

[`week9_qwen_lora_qlora_ptq_gguf.ipynb.ipynb`](./week9_qwen_lora_qlora_ptq_gguf.ipynb.ipynb)를 Google Colab에서 열고 위에서부터 순서대로 실행하세요.

1. Colab에서 `파일 > 노트 업로드`를 선택하고 위 파일을 업로드합니다.
2. `런타임 > 런타임 유형 변경`에서 `G4 GPU`를 선택합니다.
3. `런타임 > 모두 실행`을 누릅니다. 중간 셀을 건너뛰지 마세요.
4. 마지막 셀이 만든 `/content/week9_submission.zip`을 Colab 왼쪽 파일 패널에서 다운로드합니다.

노트북은 아래 과제를 모두 수행합니다.

- `Qwen/Qwen2.5-0.5B-Instruct`에 LoRA Fine-Tuning
- 같은 설정의 QLoRA Fine-Tuning 및 GPU 메모리 비교
- QLoRA adapter 병합 후 F16 GGUF 생성
- `Q4_K_M` Post-Training Quantization 적용
- llama.cpp의 F16/Q4_K_M 속도, 메모리, 답변 품질 비교
- 제출용 결과 ZIP 자동 생성

실행 결과는 Colab 세션의 `/content/week9_outputs`에, 제출 묶음은 `/content/week9_submission.zip`에 저장됩니다. Colab 세션이 종료되면 파일이 사라지므로 마지막 셀 실행 후 ZIP을 바로 다운로드하세요.

## 회고

LoRA와 QLoRA를 비교하며 학습 방식에 따라 GPU 메모리 사용량이 달라짐을 확인했다. GGUF 변환과 양자화까지 해보니 배포할 때는 성능과 용량을 함께 고려해야 했다.
