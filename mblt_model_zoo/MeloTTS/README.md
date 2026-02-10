# MeloTTS for `mblt-model-zoo`

## Introduction

MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai).
`mblt-model-zoo` provides pre-quantized version of MeloTTS models.

Currently supported languages include:

| Language | Link |
| --- | --- |
| English (American)    | [Link](https://huggingface.co/mobilint/MeloTTS-English-v3) |
| Korean                | [Link](https://huggingface.co/mobilint/MeloTTS-Korean) |

## Usage

### Installation notes

`mblt-model-zoo` exposes MeloTTS as an optional extra:

```bash
pip install "mblt-model-zoo[MeloTTS]"
```

The `MeloTTS` extra installs `unidic` and `nltk`. `unidic` requires downloading its dictionary once. Also, `nltk` requires downloading its resource once.
We prepared script for downloading needed files for `MeloTTS`.

```bash
mblt-melotts-download
```

### WebUI

The WebUI supports muliple languages and voices. First, follow the installation steps. Then, simply run:

```bash
mblt-model-zoo melo-ui
# Or: python melo/app.py
```

### CLI

You may use the MeloTTS CLI to interact with MeloTTS. The CLI may be invoked using either `melotts` or `melo`. Here are some examples:

**Read English text:**

```bash
mblt-model-zoo melo "Text to read" output.wav
```

**Specify a language:**

```bash
mblt-model-zoo melo "Text to read" output.wav --language EN_NEWEST
```

**Specify a speed:**

```bash
mblt-model-zoo melo "Text to read" output.wav --language EN_NEWEST --speed 1.5
mblt-model-zoo melo "Text to read" output.wav --speed 1.5
```

**Use a different language:**

```bash
mblt-model-zoo melo "text-to-speech 안녕하세요" kr.wav -l KR
```

**Load from a file:**

```bash
mblt-model-zoo melo file.txt out.wav --file
```

The full API documentation may be found using:

```bash
mblt-model-zoo melo --help
```

### Python API

#### English

```python
from mblt_model_zoo.MeloTTS.api import TTS

# Speed is adjustable
speed = 1.0

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN_NEWEST', device='cpu')
speaker_ids = model.hps.data.spk2id

output_path = 'en-us.wav'
model.tts_to_file(text, speaker_ids['EN-Newest'], output_path, speed=speed)
```

#### Korean

```python
from mblt_model_zoo.MeloTTS.api import TTS

# Speed is adjustable
speed = 1.0

text = "안녕하세요! 오늘은 날씨가 정말 좋네요."
model = TTS(language='KR', device='cpu')
speaker_ids = model.hps.data.spk2id

output_path = 'kr.wav'
model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)
```

## Original Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This part of package is under MIT License, which means it is free for both commercial and non-commercial use.
