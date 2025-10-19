// web_app/static/js/voice.js
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null; // fallback only
let workletNode = null;   // preferred path
let recording = false;
let chunks = [];
let inputSampleRate = 48000; // default; will update from AudioContext

let ws = null;
let streamBuffer = [];  // Float32 chunks awaiting framing
let streamBytesPerFrame = 0;
let streamSampleRate = 48000;
let streaming = false;

function float32ToPCM16(float32) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        let s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
}

// Accumulate chunks to fixed 20ms frames at input SR (typically 48k)
function frameAndSendIfReady() {
    const needed = streamBytesPerFrame; // samples
    let total = 0;
    for (const b of streamBuffer) total += b.length;
    if (total < needed) return;

    // Collect exactly 'needed' samples
    const frame = new Float32Array(needed);
    let filled = 0;
    while (filled < needed) {
        const head = streamBuffer[0];
        const remaining = needed - filled;
        if (head.length <= remaining) {
            frame.set(head, filled);
            filled += head.length;
            streamBuffer.shift();
        } else {
            frame.set(head.subarray(0, remaining), filled);
            streamBuffer[0] = head.subarray(remaining);
            filled = needed;
        }
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
        const pcm16 = float32ToPCM16(frame);
        ws.send(pcm16.buffer);
    }
}


let rollingBuffer = [];

let wakeSession = {
    armed: false,
    assembled: '',
    debounce: null
};

let llmBusy = false;
const pendingQueries = [];

async function drainQueue() {
    if (llmBusy) return;
    const next = pendingQueries.shift();
    if (!next) return;

    llmBusy = true;
    try {
        const inputEl = document.getElementById('user-input');
        if (inputEl) inputEl.value = next;
        // Uses the existing chat flow
        await requestFormattedPrompt();
    } catch (e) {
        console.error('LLM dispatch failed:', e);
    } finally {
        llmBusy = false;
        if (pendingQueries.length) await drainQueue();
    }
}

async function enqueueQuery(q) {
    pendingQueries.push(q);
    if (!llmBusy) await drainQueue();
}

function finalizeUtterance() {
    wakeSession.debounce = null;
    const finalText = wakeSession.assembled.trim();
    wakeSession.assembled = '';
    wakeSession.armed = false;
    if (finalText) enqueueQuery(finalText);
}

function clearWakeDebounce() {
    if (wakeSession.debounce) {
        clearTimeout(wakeSession.debounce);     // The clearTimeout() function is used to clear a timeout previously set with setTimeout().
        wakeSession.debounce = null;
    }
}

function resetWakeDebounce() {
    clearWakeDebounce();
    // Silence window that ends the utterance (tune 500-1200ms as needed)
    wakeSession.debounce = setTimeout(finalizeUtterance, 700);
}

function cancelIfStop(text) {
    const low = String(text || '').toLowerCase();
    if (low.includes('cancel') || low.includes('never mind') || low.includes('stop')) {
        clearWakeDebounce();
        wakeSession.armed = false;
        wakeSession.assembled = '';
        return true;
    }
    return false;
}

function stripWakeWord(text) {
    let wakeWord = document.getElementById('asr_wake_word').value;
    console.log('stripWakeWord received text:', text);
    const t = String(text || '');
    const idx = t.toLowerCase().indexOf(wakeWord);
    if (idx === -1) return t.trim();    // idx === -1 means the wake word is not found in the text
    console.log('stripWakeWord found wakeWord at index:', idx);
    // Remove the first occurrence of the wake word and trim punctuation/space around it
    const before = t.slice(0, idx).trim();    // before is the text before the wake word
    const after = t.slice(idx + wakeWord.length).trim();    // after is the text after the wake word
    return (before ? before + ' ' : '') + after;    // return the text before the wake word and the text after the wake word
}

async function determineLlmUse(transcription) {
    let wakeWord = document.getElementById('asr_wake_word').value;
    const text = String(transcription || '').trim();
    if (!text) return;

    // Maintain last 100 raw fragments for debugging/context
    rollingBuffer.push(text);
    if (rollingBuffer.length > 100) rollingBuffer.shift();  // The shift() method removes the first element of the array and returns it.

    // Optional stop/cancel handling (works both pre and post wake)
    if (cancelIfStop(text)) return;

    if (!wakeSession.armed) {
        // Wait for wake word
        if (text.toLowerCase().includes(wakeWord)) {
            // Use raw text as is, no need to strip wake word for now
            const cleaned = stripWakeWord(text);
            console.log('cleaned:', cleaned);
            wakeSession.armed = true;
            wakeSession.assembled = cleaned;
            resetWakeDebounce();
        }
        return;
    }

    // Already armed: accumulate and debounce to detect end of utterance
    wakeSession.assembled = (wakeSession.assembled ? (wakeSession.assembled + ' ') : '') + text;
    resetWakeDebounce();
}


async function startStreamingASR() {
    if (streaming) return;
    streaming = true;

    // Start mic capture (reusing your existing startRecording graph)
    await startRecording();

    // Open WebSocket to the sidecar ASGI server
    // Example: ws://localhost:9070/ws/asr (adjust host/port) 'ws://localhost:10087/ws/asr'
    const asrWsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + getHfwAsrAsgiHost() + ':' + getHfwAsrAsgiPort() + '/ws/asr';
    console.log('asrWsUrl:', asrWsUrl);

    // Create a URLSearchParams object to build the query string
    const queryParams = new URLSearchParams();
    queryParams.append('source_samplerate', inputSampleRate || 48000);

    // Define the element IDs for the parameters
    const paramIds = [
        'asr_temperature',
        'asr_max_new_tokens',
        'asr_samplerate',
        'asr_volume_threshold',
        'asr_silence_duration_s',
        'asr_min_chunk_duration_s',
        'asr_min_context_s',
        'asr_stale_buffer_timeout_s',
        'asr_min_meaningful_samples_factor',
        'asr_vad_threshold',
        'asr_vad_min_speech_ms',
        'asr_vad_min_silence_ms',
        'asr_vad_window_size_samples',
        'asr_vad_max_buffer_s',
        'asr_vad_speech_pad_ms'
    ];

    // Add checkbox element IDs
    const checkboxIds = [
        'asr_apply_normalization',
        'asr_apply_tts_padding',
        'asr_apply_zero_padding',
        'asr_apply_rms_dimming',
        'asr_apply_crossfade'
    ];

    // Append parameters from input fields
    paramIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            queryParams.append(id, element.value);
        }
    });

    // Append parameters from checkboxes
    checkboxIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            queryParams.append(id, element.checked);
        }
    });

    // Construct the final WebSocket URL with query parameters
    const asrWsUrlWithQueryParams = asrWsUrl + '?' + queryParams.toString();
    console.log('asrWsUrlWithQueryParams:', asrWsUrlWithQueryParams);

    ws = new WebSocket(asrWsUrlWithQueryParams);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        // Prepare 20ms frames at source SR
        streamSampleRate = inputSampleRate || 48000;
        streamBytesPerFrame = Math.round(0.02 * streamSampleRate); // 20ms
    };

    ws.onmessage = (evt) => {
        try {
            const msg = JSON.parse(typeof evt.data === 'string' ? evt.data : '');
            if (msg.type === 'transcript' && msg.text) {
                determineLlmUse(msg.text);
            }
        } catch {
            // ignore binary or unexpected messages
        }
    };

    ws.onclose = () => { ws = null; };
    ws.onerror = () => { /* optional logging */ };

    // Hook worklet delivery into the streaming buffer
    // We reuse your existing workletNode/processorNode paths:
    const origWorkletHandler = workletNode?.port.onmessage;
    if (workletNode) {
        workletNode.port.onmessage = (e) => {
            if (!streaming) return;
            const data = e.data;
            let chunk = null;
            if (data instanceof ArrayBuffer) chunk = new Float32Array(data);
            else if (data && data.buffer) chunk = new Float32Array(data.buffer);
            if (chunk) {
                streamBuffer.push(chunk);
                frameAndSendIfReady();
            }
            if (origWorkletHandler && origWorkletHandler !== workletNode.port.onmessage) {
                origWorkletHandler(e);
            }
        };
    } else if (processorNode) {
        const origProc = processorNode.onaudioprocess;
        processorNode.onaudioprocess = (e2) => {
            if (streaming) {
                const input = e2.inputBuffer.getChannelData(0);
                streamBuffer.push(new Float32Array(input));
                frameAndSendIfReady();
            }
            if (origProc && origProc !== processorNode.onaudioprocess) {
                origProc(e2);
            }
        };
    }
}


function stopStreamingASR() {
    if (!streaming) return;
    streaming = false;

    if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.close(); } catch (_) {}
    }
    ws = null;
    streamBuffer = [];

    // Stop mic capture graph (reuses your stopRecording)
    stopRecording();
}


// Optional UI wiring
async function toggleStreaming(btnEl) {
    if (!streaming) {
        await startStreamingASR();
        btnEl.querySelector('i').classList.remove('fa-microphone-slash');
        btnEl.querySelector('i').classList.add('fa-microphone'); 
    } else {
        stopStreamingASR();
        btnEl.querySelector('i').classList.remove('fa-microphone');
        btnEl.querySelector('i').classList.add('fa-microphone-slash');
    }
}

window.__voice_streaming__ = { startStreamingASR, stopStreamingASR, toggleStreaming };  // onclick() event for button with ID recordBtn in chat.html!


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

    // // Merge Float32 chunks
    // const merged = mergeFloat32(chunks);
    // // Resample to 16kHz
    // const resampled = resampleFloat32(merged, inputSampleRate, 16000);
    // // Encode WAV (PCM16, mono, 16kHz)
    // const wavBlob = encodeWavPCM16(resampled, 16000);

    // // Send to backend
    // uploadWav(wavBlob);
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

        const asrWaitress_URL = getHfwAsrUrl()
        
        headers = {
            'Content-Type': 'multipart/form-data',
            'X-ASR-Temperature': document.getElementById('asr_temperature').value,
            'X-ASR-Max-New-Tokens': document.getElementById('asr_max_new_tokens').value,
            'X-ASR-Samplerate': document.getElementById('asr_samplerate').value,
            'X-ASR-Volume-Threshold': document.getElementById('asr_volume_threshold').value,
            'X-ASR-Silence-Duration-S': document.getElementById('asr_silence_duration_s').value,
            'X-ASR-Min-Chunk-Duration-S': document.getElementById('asr_min_chunk_duration_s').value,
            'X-ASR-Min-Context-S': document.getElementById('asr_min_context_s').value,
            'X-ASR-Stale-Buffer-Timeout-S': document.getElementById('asr_stale_buffer_timeout_s').value,
            'X-ASR-Min-Meaningful-Samples-Factor': document.getElementById('asr_min_meaningful_samples_factor').value,
            'X-ASR-VAD-Threshold': document.getElementById('asr_vad_threshold').value,
            'X-ASR-VAD-Min-Speech-MS': document.getElementById('asr_vad_min_speech_ms').value,
            'X-ASR-VAD-Min-Silence-MS': document.getElementById('asr_vad_min_silence_ms').value,
            'X-ASR-VAD-Window-Size-Samples': document.getElementById('asr_vad_window_size_samples').value,
            'X-ASR-VAD-Max-Buffer-S': document.getElementById('asr_vad_max_buffer_s').value,
            'X-ASR-VAD-Speech-Pad-MS': document.getElementById('asr_vad_speech_pad_ms').value,
            'X-ASR-Apply-Normalization': document.getElementById('asr_apply_normalization').checked,
            'X-ASR-Apply-TTS-Padding': document.getElementById('asr_apply_tts_padding').checked,
            'X-ASR-Apply-Zero-Padding': document.getElementById('asr_apply_zero_padding').checked,
            'X-ASR-Apply-RMS-Dimming': document.getElementById('asr_apply_rms_dimming').checked,
            'X-ASR-Apply-Crossfade': document.getElementById('asr_apply_crossfade').checked,
        };

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

//window.__voice_recorder__ = { startRecording, stopRecording, toggleRecord };  // onclick() event for button with ID recordBtn in chat.html!