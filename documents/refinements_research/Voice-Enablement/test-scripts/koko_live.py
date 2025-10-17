from kokoro import KPipeline
import torch
import io
import sys
import soundfile as sf

try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

pipeline = KPipeline(lang_code='a')
text = '''
Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
'''
generator = pipeline(text, voice='af_sky')

for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    if HAVE_SD:
        sd.play(audio, samplerate=24000)
        sd.wait()  # block until finished
    else:
        if sys.platform == "win32":
            import winsound
            buf = io.BytesIO()  # creat & write to in-memory buffer
            sf.write(buf, audio, 24000, format='WAV', subtype='PCM_16')
            winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)
        else:
            sf.write(f'{i}.wav', audio, 24000)