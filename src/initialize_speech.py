import soundfile as sf
import numpy as np
import torch
from src.voice_utils import log_mel_feature
from transformers import WhisperProcessor,WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


def get_text_from_audio():
    audio_data,sr = sf.read(file= "output.wav")
    #input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
    input_features = log_mel_feature(audio=audio_data,sr=sr)
    input_features = np.expand_dims(input_features,0)
    input_features = input_features.astype(np.float32)
    input_features = torch.from_numpy(input_features) 
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription, audio_data



def initialize_voice():
    command = None
    print("Recording...")
    command,audio_data = get_text_from_audio()
    return command[0], audio_data





