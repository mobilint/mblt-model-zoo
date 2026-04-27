"""Batch tests for HyperCLOVA X models."""

MODEL_PATHS = ("mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B-Batch16",)


def test_clova_batch(pipe, run_batch_generation) -> None:
    """Run short mixed-language prompts against HyperCLOVA X batch models."""
    system_prompt = (
        '- AI model name is "CLOVA X", created by NAVER.\n'
        "- Answer briefly and helpfully.\n"
        "- You can answer in Korean or English depending on the question."
    )
    messages = [
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Explain gravity in one sentence."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "한강을 짧게 소개해줘."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Give one study tip for exams."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "비 오는 날 듣기 좋은 음악 분위기 추천해줘."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is an algorithm?"},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "면접 시작 인사 한 문장 써줘."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Summarize cloud computing simply."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "집중력 높이는 습관 하나만 말해줘."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Name one benefit of exercise."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "한국어로 API를 짧게 설명해줘."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Give a short product slogan."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "오늘 할 일 3개 정리하는 방법 알려줘."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is open source?"},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "부산 여행 포인트 두 개만 추천해줘."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a polite thank-you sentence."},
        ],
        [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "영어 발표할 때 떨림 줄이는 팁 하나 알려줘."},
        ],
    ]
    run_batch_generation(pipe, messages)
