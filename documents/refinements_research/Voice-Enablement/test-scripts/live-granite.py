from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from nemo.collections.speechlm2.models import SALM
from scipy.io.wavfile import read   # To read the in-memory audio wav file
from scipy.io.wavfile import write as wav_write

import sounddevice as sd
import numpy as np
import torchaudio
import threading
import tempfile
import pyttsx3
import librosa
import torch
import queue
import time
import os
import io   # For in-memory audio file


####---GLobals!---####

# --- Real-time Audio Processing ---
audio_queue = queue.Queue()
is_recording = threading.Event()
samplerate = 16000  # Whisper expects 16kHz

# Config for Intelligent Padding:
VOLUME_THRESHOLD = 0.04  # Adjust this threshold based on your environment
SILENCE_DURATION_S = 1.5  # Duration of silence to consider as a pause
MIN_CHUNK_DURATION_S = 0.25  # Minimum duration of a chunk to consider it for processing
MIN_CONTEXT_S = 11 # If audio is shorter than this, we'll pad it
STALE_BUFFER_TIMEOUT_S = 20.0   # How long to wait before checking for staleness.
MIN_MEANINGFUL_SAMPLES = 1.5 * samplerate  # Max samples to be considered "stale noise". 16000 samples = 1 second every STALE_BUFFER_TIMEOUT_S can be considered stale, any more and it must be processed.

# The nonsensical phrase to pad with. Make it unique!
PADDING_TEXT = " tony is quiet silent for too long I must not keep master waiting bad dooby must obey and transcribe dooby good servant will transcribe otherwise I will be severely punished"

### VAD Stuff:
### Load Silero VAD
VAD_DEVICE = "cpu"  # keep VAD on CPU to leave GPU for ASR
silero_vad, silero_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = silero_utils
silero_vad.to(VAD_DEVICE).eval()

# VAR tuning for noisy / far-field:
VAD_THRESHOLD = 0.5                 # higher => stricter speech acceptance (0.5–0.6 is typical in noise)
VAD_MIN_SPEECH_MS = 250             # avoid ultra-short blips
VAD_MIN_SILENCE_MS = 500            # hangover; end utterance after ~0.5s silence
VAD_WINDOW_SIZE_SAMPLES = 1536      # 96 ms @ 16 kHz (Silero default/robust pick)
MAX_BUFFER_S = 30                  # cap rolling buffer to 30s
VAD_SPEECH_PAD_MS = 30

APPLY_NORMALIZATION = True
APPLY_TTS_PADDING = True
APPLY_ZERO_PADDING = False
APPLY_RMS_DIMMING = True
APPLY_CROSSFADE = False

#####-------------#####


def get_indices_of_substring(response, start_substring, end_substring):
    print("\nAttempting to trim response...\n")
    try:
        if start_substring in response and end_substring in response:
            start_index = response.rindex(start_substring)  # Sometimes the model re-gurgitates multiple copies of the same dict in it's response
            end_index = response.rindex(end_substring) # rindex() returns the index of the last occurrence of the substring
            print("\nSubstring successfully found, returning indices...\n")
            return start_index, (end_index + len(end_substring))
            
        else:
            print(f"\nResponse does not contain either the start_substring: {start_substring} or the end_substring: {end_substring}, returning unchanged response...\n")
            return None, None
    except Exception as e:
        print(f"Failed to trim response, encountered error: {e}")
        return None, None


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ibm-granite/granite-speech-3.3-8b"
# MODEL_ID = "ibm-granite/granite-speech-3.3-2b"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
TOKENIZER = PROCESSOR.tokenizer
MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16
)
MODEL.to(DEVICE)


def generate_padding_audio(text:str, sr:int=16000) -> np.array:
    """Generates audio for padding text using pyttsx3 and returns it as a NumPy array."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of the speech - Adjust as needed
    engine.save_to_file(text, 'temp_padding.wav')   # pyttsx3 can be tricky with in-memory, so a temp file is robust
    engine.runAndWait()

    # Read the wav file from disk and convert to the correct format
    # The sample rate for pyttsx3 might not be 16000, so we'll need to resample later if needed
    read_sr, audio_data = read('temp_padding.wav')

    # Convert to mono float32, which is what Whisper expects
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)    # Convert to mono

    if read_sr != sr:   # resample to 16000:
        audio_data = librosa.resample(audio_data, orig_sr=read_sr, target_sr=sr)

    # This ensures the audio uses the full dynamic range from -1.0 to 1.0
    peak_volume = np.max(np.abs(audio_data))
    if peak_volume > 0:
        audio_data = audio_data / peak_volume
    
    # Clean up the temp file
    # os.remove('temp_padding.wav')
    return audio_data

#JUNK_PHRASES = ["Thank you.", "Thank you for watching!", "1 tbs of butter", "Teksting av Nicolai Winther", "1 tsk vaniljenavsak"]
JUNK_PHRASES = ["Thank you."]

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def start_recording():
    is_recording.set()
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback, dtype='float32'):
        while is_recording.is_set():
            time.sleep(0.1)     # The callback is handling the audio data, so we just wait

def stop_recording():
    is_recording.clear()


def transcribe_with_granite(audio_f32: np.ndarray, sr: int = 16000, max_new_tokens: int = 128) -> str:
    # ensure mono float32 in [-1, 1]
    if audio_f32.ndim > 1:
        audio_f32 = audio_f32.mean(axis=1)
    if sr != 16000:
        audio_f32 = librosa.resample(audio_f32, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio_f32 = np.clip(audio_f32.astype(np.float32), -1.0, 1.0)

    # Convert numpy array directly to PyTorch tensor
    # Add batch dimension: (samples,) -> (1, samples)
    wav = torch.from_numpy(audio_f32).unsqueeze(0).float()

    try:
        system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        prompt = TOKENIZER.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        model_inputs = PROCESSOR(prompt, wav, device=DEVICE, return_tensors="pt").to(DEVICE)
        model_outputs = MODEL.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)

        # Transformers includes the input IDs in the response.
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
        output_text = TOKENIZER.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        text = output_text[0].upper()
        return text
    except Exception as e:
        # Handle any processing errors
        print(f"Error during transcription: {e}")
        return ""


### NEW VAD VERSION
if __name__ == "__main__":
    print("Starting... Generating padding audio first.")
    padding_audio = generate_padding_audio(PADDING_TEXT, sr=samplerate)
    pad_rms = np.sqrt(np.mean(padding_audio**2) + 1e-12)

    print("Starting real-time transcription. Press Ctrl+C to stop.")
    
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()
    
    audio_buffer = np.array([], dtype=np.float32)
    audio_to_process = audio_buffer.copy()
    last_speech_time = time.time()
    last_buffer_clear_time = time.time()
    global_cursor = 0   # absolute sample index of audio_buffer[0]
    last_append_end_abs = 0   # absolute sample index up to which we've appended into audio_to_process

    try:
        while True:
            # Accumulate incoming audio from the callback
            while not audio_queue.empty():
                chunk = audio_queue.get()
                audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))

            # Run Silero VAD when we have at least a little audio
            # if len(audio_buffer) >= MIN_MEANINGFUL_SAMPLES:
            if len(audio_buffer) >= int(0.5 * samplerate):  # i.e. if audio_buffer has at least 0.5s of audio
                with torch.no_grad():
                    wav_t = torch.from_numpy(audio_buffer.copy()).to(VAD_DEVICE)
                    segments = get_speech_timestamps(
                        wav_t,
                        silero_vad,
                        sampling_rate=samplerate,
                        threshold=VAD_THRESHOLD,
                        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
                        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                        window_size_samples=VAD_WINDOW_SIZE_SAMPLES,
                        speech_pad_ms=VAD_SPEECH_PAD_MS,
                    )

                if segments:
                    last_speech_time = time.time()

                last_end = 0
                for segment in segments:    # we know segments contains speech only, so update the buffers!
                    start, end = segment["start"], segment["end"]
                    abs_start = global_cursor + start
                    abs_end = global_cursor + end

                    # Clip overlap so we only append never-seen samples
                    append_start_abs = max(abs_start, last_append_end_abs)
                    if append_start_abs < abs_end:
                        relative_start = int(append_start_abs - global_cursor)
                        relative_end = int(abs_end - global_cursor)
                        new_chunk = audio_buffer[relative_start:relative_end].copy()
                        audio_to_process = new_chunk if len(audio_to_process) == 0 else np.concatenate((audio_to_process, new_chunk))
                        last_append_end_abs = abs_end

                    if end > last_end:
                        last_end = end

                # Rolling buffer trim - simplest: no overlap
                global_cursor += last_end   # remove the tail to avoid reprocessing
                if len(audio_buffer) > MAX_BUFFER_S * samplerate:
                    audio_buffer = np.array([], dtype=np.float32)
                else:
                    audio_buffer = audio_buffer[last_end:].copy()

                current_time = time.time()
                if len(audio_to_process) >= MIN_CHUNK_DURATION_S * samplerate and (current_time - last_speech_time) > SILENCE_DURATION_S:
                    print(f"Processing {len(audio_to_process)/samplerate:.2f}s of audio...")

                    if APPLY_NORMALIZATION:
                        print(f"Applying normalization to audio.")
                        peak_volume = np.max(np.abs(audio_to_process)) if len(audio_to_process) > 0 else 0.0
                        if peak_volume > 0: # normalize the spoken audio so it's not lost to the padding audio!
                            audio_to_process = audio_to_process / peak_volume

                    if APPLY_TTS_PADDING:
                        padding_applied = False
                        if len(audio_to_process) < MIN_CONTEXT_S * samplerate:
                            print(f"Audio is shorter than {MIN_CONTEXT_S}s. Applying padding.")
                            
                            if APPLY_RMS_DIMMING:
                                print(f"Applying RMS dimming to padding audio.")
                                # Set padding RMS ~20–30 dB below speech RMS and optionally low‑pass it to reduce lexical cues:
                                speech = audio_to_process
                                speech_rms = np.sqrt(np.mean(speech**2) + 1e-12)
                                target_pad_rms = speech_rms * (10 ** (-24/20))  # -24 dB
                                if pad_rms > 0:
                                    padding_audio_dimmed_rms = padding_audio * (target_pad_rms / pad_rms)
                                else:
                                    padding_audio_dimmed_rms = padding_audio

                                if APPLY_CROSSFADE:
                                    print(f"Applying crossfade to padding audio.")
                                    # short crossfade to avoid a hard boundary
                                    xf = int(0.05 * samplerate)  # 50 ms
                                    if len(speech) >= xf and len(padding_audio_dimmed_rms) >= xf:
                                        fade = np.linspace(1.0, 0.0, xf, dtype=np.float32)
                                        audio_to_process[-xf:] *= fade
                                        padding_audio_dimmed_rms[:xf] *= (1.0 - fade)
                                
                                audio_to_process = np.concatenate((audio_to_process, padding_audio_dimmed_rms))
                                padding_applied = True

                            else:
                                print(f"Applying padding audio without RMS dimming.")
                                audio_to_process = np.concatenate((audio_to_process, padding_audio))
                                padding_applied = True

                    if APPLY_ZERO_PADDING:
                        # Zero-padding as an alternative to TTS padding
                        needed = max(0, int(MIN_CONTEXT_S * samplerate - len(audio_to_process)))
                        if needed > 0:
                            print(f"Audio is shorter than {MIN_CONTEXT_S}s. Applying zero-padding.")
                            audio_to_process = np.concatenate((audio_to_process, np.zeros(needed, dtype=np.float32)))

                    transcription = transcribe_with_granite(
                        audio_to_process,
                        sr=samplerate,
                        max_new_tokens=1500,
                    )
                    print(f"\n\nRAW Transcription: {transcription}\n\n")

                    if padding_applied and transcription:
                        try:
                            padding_start_index, padding_end_index = get_indices_of_substring(
                                transcription.lower().strip(),
                                start_substring="tony is quiet",
                                end_substring="will be severely punished"
                            )
                            
                            if padding_start_index is None or padding_end_index is None:    # try once again with a small tweak!
                                padding_start_index, padding_end_index = get_indices_of_substring(
                                    transcription.lower().strip(),
                                    start_substring="tony is quite",
                                    end_substring="will be severely punished"
                                )
                            
                            if padding_start_index is not None and padding_end_index is not None:
                                transcription = transcription[:padding_start_index] + transcription[padding_end_index:]
                                transcription = transcription.replace(" .", "").strip()
                        except Exception as e:
                            print(f"Failed to trim response, encountered error: {e}")
                    
                    if transcription:
                        print(f"\n\nTranscription: {transcription}\n\n")
                    
                    # After a SUCCESSFUL transcription, clear the buffer!
                    audio_to_process = np.array([], dtype=np.float32)
                    last_append_end_abs = global_cursor
                    last_buffer_clear_time = time.time()

            # A short sleep to prevent the loop from running too fast
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping transcription.")
        stop_recording()
        recording_thread.join()