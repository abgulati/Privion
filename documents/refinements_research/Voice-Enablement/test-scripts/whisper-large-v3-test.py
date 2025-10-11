import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
# import os

# ffmpeg_bin_path = r"M:\\Storage\\Software\\ffmpeg-7.1.1-full_build\\ffmpeg-7.1.1-full_build\\bin" # Use the exact path where you installed FFmpeg

# if os.path.isdir(ffmpeg_bin_path) and hasattr(os, 'add_dll_directory'):
#     print(f"Adding FFmpeg DLL directory to path: {ffmpeg_bin_path}")
#     os.add_dll_directory(ffmpeg_bin_path)
# else:
#     print("Warning: FFmpeg DLL path not found or os.add_dll_directory is not available.")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample["array"], return_timestamps=True)
print(result["text"])
