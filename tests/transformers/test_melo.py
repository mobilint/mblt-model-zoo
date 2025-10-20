from mblt_model_zoo.transformers.text_to_speech.MeloTTS import TTS

# Speed is adjustable
speed = 1.0

device = 'auto'

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN_NEWEST', device=device)
speaker_ids = model.hps.data.spk2id

# American accent
output_path = 'en-us.wav'
model.tts_to_file(text, speaker_ids['EN-Newest'], output_path, speed=speed)
