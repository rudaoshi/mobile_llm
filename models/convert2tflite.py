
import whisper
import torch

import torch.nn as nn
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.train_util import DLContext, get_device, train

model = whisper.load_model("tiny")
model.eval()
audio = whisper.load_audio("./filename.mp3")
audio = whisper.pad_or_trim(audio)
print(audio.shape)
# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
print(mel.shape)
quantizer = PostQuantizer(model, mel, work_dir='out', config={'asymmetric': True, 'per_tensor': False})
# qat_model = quantizer.quantize()