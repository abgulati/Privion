from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import read   # To read the in-memory audio wav file

import sounddevice as sd
import numpy as np
import threading
import pyttsx3
import librosa
import torch
import queue
import time
import os
import io   # For in-memory audio file


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
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "openai/whisper-large-v3"
MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
)
MODEL.to(DEVICE)

PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)

def get_pipe():
    return pipeline(
        "automatic-speech-recognition",
        model=MODEL,
        tokenizer=PROCESSOR.tokenizer,
        feature_extractor=PROCESSOR.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
    )

# --- Real-time Audio Processing ---
audio_queue = queue.Queue()
is_recording = threading.Event()
samplerate = 16000  # Whisper expects 16kHz

# Config for Intelligent Padding:
VOLUME_THRESHOLD = 0.04  # Adjust this threshold based on your environment
SILENCE_DURATION_S = 1.5  # Duration of silence to consider as a pause
MIN_CHUNK_DURATION_S = 0.25  # Minimum duration of a chunk to consider for padding
MIN_CONTEXT_S = 11 # If audio is shorter than this, we'll pad it
STALE_BUFFER_TIMEOUT_S = 20.0   # How long to wait before checking for staleness.
MIN_MEANINGFUL_SAMPLES = 1 * samplerate  # Max samples to be considered "stale noise". 16000 samples = 1 second every STALE_BUFFER_TIMEOUT_S can be considered stale, any more and it must be processed.

# The nonsensical phrase to pad with. Make it unique!
PADDING_TEXT = " tony is quiet silent for too long I must not keep master waiting bad dooby must obey and transcribe dooby good servant will transcribe otherwise I will be severely punished"

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


if __name__ == "__main__":
    print("Starting... Generating padding audio first.")
    padding_audio = generate_padding_audio(PADDING_TEXT, sr=samplerate)

    print("Starting real-time transcription. Press Ctrl+C to stop.")
    
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()
    
    audio_buffer = np.array([], dtype=np.float32)
    last_speech_time = time.time()
    last_buffer_clear_time = time.time()

    try:
        while True:
            is_speech = False
            while not audio_queue.empty():
                chunk = audio_queue.get()
                volume_rms = np.sqrt(np.mean(chunk**2)) # Calculate the volume (Root Mean Square) of the chunk
                # print(f"RMS: {volume_rms:.4f}") # Print the volume level for debugging  - see the RMS values of your speech versus your room's silence to find the perfect value!
                
                if volume_rms > VOLUME_THRESHOLD:   # Only accumulate if above the volume threshold
                    # print("concatenated")
                    audio_buffer = np.concatenate((audio_buffer, chunk.flatten()))
                    is_speech = True
            
            if is_speech:
                last_speech_time = time.time()

            current_time = time.time()
            # The trigger condition: buffer has content AND it's been quiet for a while
            if len(audio_buffer) > 0 and (current_time - last_speech_time) > SILENCE_DURATION_S:

                if len(audio_buffer) >= MIN_CHUNK_DURATION_S * samplerate:
                    print(f"Processing {len(audio_buffer)/samplerate:.2f}s of audio...")

                    peak_volume = np.max(np.abs(audio_buffer))
                    if peak_volume > 0: # normalize the spoken audio so it's not lost to the padding audio!
                        audio_buffer = audio_buffer / peak_volume

                    audio_to_process = audio_buffer.copy()
                    padding_applied = False
                    
                    if len(audio_to_process) < MIN_CONTEXT_S * samplerate:
                        print(f"Audio is shorter than {MIN_CONTEXT_S}s. Applying padding.")
                        audio_to_process = np.concatenate((audio_to_process, padding_audio))
                        padding_applied = True

                    pipe = get_pipe()  # Get a fresh pipeline instance to avoid potential memory issues
                    
                    # Process the accumulated audio data
                    result = pipe(audio_to_process, return_timestamps=True)
                    print(f"\nRAW Result: {result}\n")
                    transcription = result["text"].strip() if result else ""

                    if padding_applied:
                        try:
                            #print(f"\ntranscription: {transcription}")
                            padding_start_index, padding_end_index = get_indices_of_substring(transcription.lower().strip(), start_substring="tony is quiet", end_substring="will be severely punished")
                            transcription = transcription[:padding_start_index] + transcription[padding_end_index:]
                            transcription = transcription.replace(" .", "").strip()
                            #print(f"\ntranscription after trimming: {transcription}")
                        except Exception as e:
                            print(f"Failed to trim response, encountered error: {e}")
                    
                    # Print the transcribed text if it's not empty
                    if transcription:
                        print(f"\n\nTranscription: {transcription}\n\n")
                
                    # After a SUCCESSFUL transcription, clear the buffer and reset the cleanup clock.
                    audio_buffer = np.array([], dtype=np.float32)
                    last_buffer_clear_time = time.time()

            # print(f"current_time - last_buffer_clear_time: {current_time - last_buffer_clear_time}")
            if (current_time - last_buffer_clear_time) > STALE_BUFFER_TIMEOUT_S and 0 < len(audio_buffer) < MIN_MEANINGFUL_SAMPLES:
                '''
                Clear the buffer and reset the cleanup clock, if there isn't at least 1 second of audio every STALE_BUFFER_TIMEOUT_S (eg 20 secs), then discard the buffer.
                `MIN_MEANINGFUL_SAMPLES` acts as a safety net: if a long sentence is being spoken, the buffer won't simply force-clear every STALE_BUFFER_TIMEOUT_S secs!
                '''
                print(f"\nDiscarding stale noise buffer of {len(audio_buffer)/samplerate:.2f}s (less than the {MIN_MEANINGFUL_SAMPLES/samplerate:.1f}s meaningful threshold).\n")
                audio_buffer = np.array([], dtype=np.float32)
                last_buffer_clear_time = time.time()
            
            # A short sleep to prevent the loop from running too fast
            time.sleep(0.05)


    except KeyboardInterrupt:
        print("\nStopping transcription.")
        stop_recording()
        recording_thread.join()