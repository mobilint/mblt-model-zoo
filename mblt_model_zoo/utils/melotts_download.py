from __future__ import annotations

import subprocess
import sys

def main() -> int:
    try:
        import nltk
        nltk.download('averaged_perceptron_tagger_eng')
    except Exception as e:  # pragma: no cover
        print(
            "nltk is not installed. Install MeloTTS extras first, e.g.\n"
            '  pip install "mblt-model-zoo[MeloTTS]"\n'
            "\n"
            "Then run python code:\n"
            "  import nltk\n",
            "  nltk.download('averaged_perceptron_tagger_eng')\n",
            file=sys.stderr,
        )
        print(f"Original error: {e}", file=sys.stderr)
        return 1
    
    try:
        import unidic  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(
            "unidic is not installed. Install MeloTTS extras first, e.g.\n"
            '  pip install "mblt-model-zoo[MeloTTS]"\n'
            "\n"
            "Then run:\n"
            "  python -m unidic download\n",
            file=sys.stderr,
        )
        print(f"Original error: {e}", file=sys.stderr)
        return 1

    return subprocess.call([sys.executable, "-m", "unidic", "download"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
