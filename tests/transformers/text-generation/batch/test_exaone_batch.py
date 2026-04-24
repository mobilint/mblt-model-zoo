"""Batch tests for EXAONE instruct models."""

MODEL_PATHS = (
    "mobilint/EXAONE-3.5-2.4B-Instruct-Batch16",
    "mobilint/EXAONE-3.5-7.8B-Instruct-Batch16",
)


def test_exaone_batch(pipe, run_batch_generation) -> None:
    """Run short multilingual prompts against EXAONE batch models."""
    messages = [
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "Explain transformers in one sentence."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "파이썬 리스트와 튜플 차이 한 줄로 설명해줘."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "Name two uses of large language models."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "서울 여행 팁 2가지만 알려줘."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "Write a short greeting email opening."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "영어 공부할 때 shadowing이 왜 좋은지 짧게 설명해줘."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "What is overfitting?"},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "간단한 아침 메뉴 하나 추천해줘."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "Give one tip for better presentations."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "회의 전에 확인할 체크포인트 두 개만 말해줘."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "Summarize recursion in plain English."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "API가 뭔지 비개발자에게 짧게 설명해줘."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "Suggest a short team motto."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "집중 안 될 때 바로 할 수 있는 방법 하나 알려줘."},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "What makes a prompt specific?"},
        ],
        [
            {"role": "system", "content": "You are EXAONE from LG AI Research. Answer briefly and clearly."},
            {"role": "user", "content": "짧은 영어 자기소개 첫 문장 예시 하나 줘."},
        ],
    ]
    run_batch_generation(pipe, messages)
