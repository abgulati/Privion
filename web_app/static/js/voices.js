// web_app/static/js/voice.js
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null; // fallback only
let workletNode = null;   // preferred path
let recording = false;
let chunks = [];
let inputSampleRate = 48000; // default; will update from AudioContext

async function startRecording() {
    if (recording) return;
    recording = true;
    chunks = [];

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStream = stream;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        inputSampleRate = audioContext.sampleRate;

        sourceNode = audioContext.createMediaStreamSource(stream);

        // Try modern AudioWorklet; fall back to ScriptProcessor only if unavailable
        try {
            // Adjust path if your static URL prefix differs
            await audioContext.audioWorklet.addModule('/static/js/recorder-worklet.js');
            workletNode = new AudioWorkletNode(audioContext, 'recorder-processor');

            workletNode.port.onmessage = (e) => {
                if (!recording) return;
                const data = e.data;
                if (data instanceof ArrayBuffer) {
                    chunks.push(new Float32Array(data));
                } else if (data && data.buffer) {
                    chunks.push(new Float32Array(data.buffer));
                }
            };

            // Keep node "live" in the graph; processor outputs silence
            sourceNode.connect(workletNode);
            workletNode.connect(audioContext.destination);
        } catch (err) {
            console.warn('AudioWorklet unavailable, falling back to ScriptProcessorNode.', err);
            // ScriptProcessorNode is deprecated but used only as a fallback
            const bufferSize = 4096;
            processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
            processorNode.onaudioprocess = e2 => {
                if (!recording) return;
                const input = e2.inputBuffer.getChannelData(0);
                chunks.push(new Float32Array(input));
            };
            sourceNode.connect(processorNode);
            processorNode.connect(audioContext.destination);
        }
    } catch (err) {
        console.error('Mic access error:', err);
        recording = false;
        alert('Microphone access was denied or failed.');
    }
}

function stopRecording() {
    if (!recording) return;
    recording = false;

    // Stop audio graph
    try {
        if (workletNode) {
            workletNode.disconnect();
            workletNode = null;
        }
        if (processorNode) {
            processorNode.disconnect();
            processorNode = null;
        }
        if (sourceNode) {
            sourceNode.disconnect();
            sourceNode = null;
        }
    } catch (e) {
        console.warn('Audio graph cleanup warning:', e);
    }

    if (audioContext) {
        // Best-effort close
        try { audioContext.close(); } catch (_) {}
        audioContext = null;
    }

    // Stop media tracks
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }

    // Merge Float32 chunks
    const merged = mergeFloat32(chunks);
    // Resample to 16kHz
    const resampled = resampleFloat32(merged, inputSampleRate, 16000);
    // Encode WAV (PCM16, mono, 16kHz)
    const wavBlob = encodeWavPCM16(resampled, 16000);

    // Send to backend
    uploadWav(wavBlob);
}

function mergeFloat32(buffers) {
    const length = buffers.reduce((acc, b) => acc + b.length, 0);
    const out = new Float32Array(length);
    let offset = 0;
    for (const b of buffers) {
        out.set(b, offset);
        offset += b.length;
    }
    return out;
}

// Simple linear resampler
function resampleFloat32(input, fromRate, toRate) {
    if (fromRate === toRate) return input;
    const ratio = toRate / fromRate;
    const newLength = Math.round(input.length * ratio);
    const output = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const inPos = i / ratio;
        const idx = Math.floor(inPos);
        const frac = inPos - idx;
        const s0 = input[idx] || 0;
        const s1 = input[idx + 1] || s0;
        output[i] = s0 + (s1 - s0) * frac;
    }
    return output;
}

function encodeWavPCM16(samplesFloat32, sampleRate) {
  // Convert float32 [-1,1] to int16
    const samples = new Int16Array(samplesFloat32.length);
    for (let i = 0; i < samplesFloat32.length; i++) {
        let s = Math.max(-1, Math.min(1, samplesFloat32[i]));
        samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);         // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);          // AudioFormat (1 = PCM)
    view.setUint16(22, 1, true);          // NumChannels
    view.setUint32(24, sampleRate, true); // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    view.setUint16(32, 2, true);          // BlockAlign (NumChannels * BitsPerSample/8)
    view.setUint16(34, 16, true);         // BitsPerSample

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM samples
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        view.setInt16(offset, samples[i], true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

async function uploadWav(wavBlob) {
    try {
        const form = new FormData();
        form.append('audio', wavBlob, 'recording.wav');

        const asrWaitress_URL = getHfwAsrUrl();
        url = `${asrWaitress_URL}/transcribe`;

        const res = await fetch(url, {
            method: 'POST',
            body: form,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            console.error('Transcribe error:', err);
            alert('Transcription failed.');
            return;
        }

        const data = await res.json();
        console.log('Transcription:', data);

        // Paste into user-input
        const userInput = document.getElementById('user-input');
        console.log('userInput.value:', userInput.value);
        userInput.value == '' ? userInput.value = data.transcription : userInput.value += ' ' + data.transcription;
        userInput.focus();
    } catch (e) {
        console.error(e);
        alert('Upload failed.');
    }
}

// Simple UI hook
function toggleRecord(btnEl) {
    if (!recording) {
        startRecording();
        btnEl.querySelector('i').classList.remove('fa-microphone');
        btnEl.querySelector('i').classList.add('fa-microphone-slash');
    } else {
        stopRecording();
        btnEl.querySelector('i').classList.remove('fa-microphone-slash');
        btnEl.querySelector('i').classList.add('fa-microphone');
    }
}

window.__voice_recorder__ = { startRecording, stopRecording, toggleRecord };