import speech_recognition as sr
from rich import print
from rich.console import Console
from rich.style import Style
from typing import Union
import time
import numpy as np
import numpy as np
import librosa
from transformers import WhisperProcessor,WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")



console = Console()
base_style = Style.parse("cyan")


def get_text_from_audio():
    audio_data = librosa.load("output.wav", sr=16000)[0]
    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription, audio_data



def initialize_voice():
    command = None
    print("Recording...")
    command,audio_data = get_text_from_audio()
    return command[0], audio_data





