"""Non-batch tests for HyperCLOVA X models."""

MODEL_PATHS = ("mobilint/HyperCLOVAX-SEED-Text-Instruct-1.5B",)


def test_clova(pipe) -> None:
    """Run a basic prompt against HyperCLOVA X."""
    pipe.generation_config.max_new_tokens = None

    messages = [
        {"role": "tool_list", "content": ""},
        {
            "role": "system",
            "content": '- AI 모델 이름은 "CLOVA X"이고 NAVER에서 만들었습니다.\n- 간결하고 정확하게 답변하세요.',
        },
        {
            "role": "user",
            "content": "뉴턴의 운동 법칙과 중력의 관계를 짧게 설명해줘.",
        },
    ]

    pipe(messages, max_length=512)
