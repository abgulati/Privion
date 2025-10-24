// web_app/static/js/voice.js

let mediaStream = null; // holds the MediaStream object that represents the audio input from the microphone
let audioContext = null; // holds the AudioContext object that represents the audio context - used to create the audio nodes and connect them to the microphone
let sourceNode = null; // holds the MediaStreamSourceNode object that represents the audio source node - used to connect the microphone to the audio context
let processorNode = null; // fallback only - the ScriptProcessorNode is deprecated but used only as a fallback
let workletNode = null;   // preferred path - the AudioWorkletNode is the preferred path for audio processing
let recording = false; // holds the boolean value that indicates whether the recording is active or not
let chunks = []; // holds the array of Float32Array objects that represent the audio chunks
let inputSampleRate = 48000; // default; will update from AudioContext

let ws = null;   // WebSocket object that holds the connection to the ASGI server
let streamBuffer = [];  // Array of Float32Array objects that represent the audio chunks awaiting framing
let streamBytesPerFrame = 0; // Number of bytes per frame - Updated in startStreaming-ASR() on WS-Open event to reflect 20ms once the final SR (below) is known
let streamSampleRate = 48000; // Sample rate of the audio stream
let streaming = false; // Boolean value that indicates whether the streaming is active or not


function float32ToPCM16(float32) {
    /*
    Function is used to convert a Float32Array object to a Int16Array object:

        - Clamp the float32 sample to the valid range [-1, 1] to prevent overflow.
        - Float32 audio can exceed this range, but 16-bit PCM is limited to [-32768, 32767].
        - Math.min(1, x) ensures x doesn't exceed 1, Math.max(-1, result) ensures it's not below -1.
        - This prevents distortion when scaling to 16-bit integer range below.
        - Convert to 16-bit PCM: 0x8000 = 32768 (min negative), 0x7FFF = 32767 (max positive)
     */
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        let s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
}


// Accumulate chunks to fixed 20ms frames at input SR (typically 48k)
function frameAndSendIfReady() {
    /*
    Function is used to frame and send the audio chunks if the WebSocket is open and ready:

        - Calculate the number of samples needed for the frame.
        - Calculate the total number of samples in the stream buffer - if less than the number of samples needed, return
        - Else - Collect exactly 'needed' samples in a frame, which is then converted to PCM16 and sent to the WebSocket server.
    */

    const needed = streamBytesPerFrame; // ~20ms - see declaration note at the top of the file
    let total = 0;
    for (const b of streamBuffer) total += b.length;
    if (total < needed) return;

    const frame = new Float32Array(needed); // Collect exactly 'needed' samples
    
    // Process chunks from streamBuffer to fill exactly 'needed' samples for the frame
    let filled = 0;
    while (filled < needed) {
        const head = streamBuffer[0];  // Get first chunk from buffer
        const remaining = needed - filled;  // How many more samples we need - init at start to know how much to take from the chunk in this iteration
        
        if (head.length <= remaining) {
            // Case 1: Entire chunk fits in remaining space - consume entire chunk, remove from buffer
            frame.set(head, filled);  // frame.set(source_data, destination_index): Copy entire chunk to frame at current position
            filled += head.length;    // Update filled count
            streamBuffer.shift();     // Remove consumed chunk from buffer
        } else {
            // Case 2: Chunk is larger than remaining space - take only what's needed, keep remainder
            frame.set(head.subarray(0, remaining), filled);  // Copy only needed portion: frame.set(source_data, destination_index)
            streamBuffer[0] = head.subarray(remaining);      // Keep leftover in buffer: subarray(start_index, end_index[Optional | Default: length of array])
            filled = needed;  // We're done filling the frame - update filled count to the total number of samples needed
        }
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
        const pcm16 = float32ToPCM16(frame);
        ws.send(pcm16.buffer);
    }
}



function stripWakeWord(text) {
    /*
    Removes a configured wake-word from the beginning (or anywhere) in a
    transcription string and cleans up surrounding punctuation / whitespace:

        - The wake-word is read live from the DOM element `#asr_wake_word`.  The
        - comparison is **case-insensitive** and only the **first occurrence** is
        - stripped.  After removal, any dangling commas or extra spaces are also
        - tidied so the resulting string is ready to be sent to the LLM.
    */

    let wakeWord = document.getElementById('asr_wake_word').value;
    console.log('strip-WakeWord received text:', text);
    const t = String(text || '');
    const idx = t.toLowerCase().indexOf(wakeWord);
    if (idx === -1) return t.trim();    // idx === -1 means the wake word is not found in the text
    console.log('strip-WakeWord found wakeWord at index:', idx);
    // Remove the first occurrence of the wake word and trim punctuation/space around it
    let before = t.slice(0, idx).trim();    // before is the text before the wake word
    let after = t.slice(idx + wakeWord.length).trim();    // after is the text after the wake word

    // if there's a preceeding or trailing comma, remove it
    if (before.endsWith(',')) before = before.slice(0, -1).trim();
    if (after.startsWith(',')) after = after.slice(1).trim();

    let final = (before ? before + ' ' : '') + after;
    // if there's any double spacing from the above removals, remove it
    final = final.replace(/  /g, ' ');
    return final;
}



let llmBusy = false;
const pendingQueries = [];

async function drainQueue() {
    /*
    Function is used to drain the pending queries queue:

        - If the LLM is busy or there are no pending queries, return.
        - Else - Set the LLM to busy.
        - Try - Transfer query to the user-input field and trigger dispatch to the LLM - catch and log any errors.
        - Finally - Set the LLM to not busy and drain the queue again if there are any pending queries.
    */

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
    pendingQueries.push(q); // push to pending Q and trigger LLM execution to drain!
    if (!llmBusy) await drainQueue();   // else if LLM is indeed busy, drainQ will handle it automatically in the finally block!
}


let wakeSession = {
    armed: false,
    assembled: '',
    debounce: null
};

function finalizeUtterance() {
    /*
    Will finalize an utterance in two steps: reset Wake-Session to default state and enqueue the query.
    */
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
    /*
    Set timeout to finalize utterance, invoked when transcription with the wake word is detected.
    Will clear the timeout on each onvocation but once the timeout expires, the utterance will 
    be finalized and the LLM will be triggered to execute the query!
    */
    clearWakeDebounce();
    wakeSession.debounce = setTimeout(finalizeUtterance, 700); // 700ms is the default silence window that ends the utterance - tune as needed
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


let rollingBuffer = []; // Used to maintain the last 100 raw fragments for debugging/context.

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
            const cleaned = stripWakeWord(text);
            console.log('cleaned:', cleaned);
            wakeSession.armed = true;
            wakeSession.assembled = cleaned;
            resetWakeDebounce();
        }
        return;
    }

    // If not returned above, that means a new WS onmessage transcription event occurred when a session was already armed:
    // In this case, we continue to accumulate transcription fragments to the ongoing session and reset debounce timer as usual.
    wakeSession.assembled = (wakeSession.assembled ? (wakeSession.assembled + ' ') : '') + text;
    resetWakeDebounce();
}


async function startStreamingASR() {
    if (streaming) return;
    streaming = true;

    // Start mic capture (reusing your existing start-Recording graph)
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
    ws.binaryType = 'arraybuffer';  // binaryType is used to specify the type of data that will be sent and received from the WebSocket.

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
    // Store the original handler to preserve any existing audio processing functionality
    const origWorkletHandler = workletNode?.port.onmessage; // if using AudioWorkletNode, use the port.onmessage event handler.
    
    if (workletNode) {
        // Override the worklet's message handler to intercept audio data for streaming
        workletNode.port.onmessage = (e) => {
            // Only process audio data if streaming is active
            if (!streaming) return;
            
            const data = e.data;
            let chunk = null;
            
            // Handle different data formats that may arrive from the AudioWorklet
            if (data instanceof ArrayBuffer) {
                // Direct ArrayBuffer from worklet - convert to Float32Array for processing
                chunk = new Float32Array(data);
            } else if (data && data.buffer) {
                // AudioWorkletMessageEvent payload wrapped as JS object with .buffer property
                // The .buffer property points to the raw ArrayBuffer of audio data
                chunk = new Float32Array(data.buffer);
            }
            
            // If we successfully extracted audio data, add it to the streaming buffer
            if (chunk) {
                streamBuffer.push(chunk); // Add audio chunk to buffer for frame assembly
                frameAndSendIfReady(); // Check if we have enough data to send a complete frame
            }
            
            // Forward the event to any previously registered handler to maintain compatibility
            // This ensures other audio processing (visualizers, level meters, etc.) continues working
            if (origWorkletHandler && origWorkletHandler !== workletNode.port.onmessage) {
                origWorkletHandler(e);
            }   // see note below
        };
    } else if (processorNode) {
        // Fallback for deprecated ScriptProcessorNode when AudioWorklet is unavailable
        const origProc = processorNode.onaudioprocess;
        
        // Override the processor's audio event handler
        processorNode.onaudioprocess = (e2) => {
            if (streaming) {
                // Extract mono audio data from the input buffer (channel 0)
                const input = e2.inputBuffer.getChannelData(0);
                // Copy the audio data and add to streaming buffer
                streamBuffer.push(new Float32Array(input));
                frameAndSendIfReady(); // Attempt to send complete frames to WebSocket
            }
            
            // Forward to original handler to preserve existing functionality
            if (origProc && origProc !== processorNode.onaudioprocess) {
                origProc(e2);
            }   // see note below
        };
    }

    /*
    NOTE ON THE origWorkletHandler / origProc FORWARDING CLAUSES:

    Placement
    ---------
    The forwarding logic  above (origWorkletHandler / origProc) is placed after the 
    WebSocket event handlers to ensure that the WebSocket connection is fully set up
    before any audio processing begins. This separation helps keep network-related code
    (WebSocket) distinct from media-related code (audio processing), making the code
    easier to follow and less error-prone.

    Purpose
    -------
    These clauses implement a *non-destructive* handler override.  This means that the 
    original handler (origWorkletHandler or origProc) is not replaced, but rather the new 
    handler (workletNode.port.onmessage or processorNode.onaudioprocess) is added to the chain.
    This is important because it allows the other parts of the application to continue to 
    receive the audio data without any interruption.

    By storing the original handler and explicitly invoking it after our own processing, 
    we guarantee that any previously-registered listeners (such as visualisers, level meters, 
    echo-cancellation, etc.) continue to receive every audio buffer. This avoids the need for 
    multiple worklets/processors (which would double CPU load and complicate device sharing) 
    and keeps the code modular — each feature can attach its own lightweight handler instead 
    of monopolising the node.

    NOTE on the nested-if checks - `origWorkletHandler !== workletNode.port.onmessage` and
    `origProc !== processorNode.onaudioprocess`:
    ------------------------------------------------------------
    They're NOT busy-loops.  The origWorklet/Proc handlers are invoked only when the underlying 
    Web Audio thread delivers a new audio buffer (typically every 128 samples ≈ 2.7 ms @ 48 kHz).  
    These calls simply forward the event to any previously-registered handler so that other 
    parts of the application can still receive the same data without creating a second worklet.
    By ensuring the inequalities, we prevent an infinite loop by confirming the original handler 
    is not the same as the one we just installed.
    */
}


function stopStreamingASR() {
    if (!streaming) return;
    
    // else close connection and restore all core variables to defaults:
    streaming = false;

    if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.close(); } catch (_) {}
    }
    ws = null;
    streamBuffer = [];

    // Stop mic capture graph
    stopRecording();
}


// Optional UI wiring
async function toggleStreaming(btnEl) {
    if (!streaming) {
        await startStreamingASR();
        btnEl.querySelector('i').classList.remove('fa-microphone-slash');
        btnEl.querySelector('i').classList.add('fa-microphone');
        document.getElementById('recordBtn').classList.add('recording');
    } else {
        stopStreamingASR();
        btnEl.querySelector('i').classList.remove('fa-microphone');
        btnEl.querySelector('i').classList.add('fa-microphone-slash');
        document.getElementById('recordBtn').classList.remove('recording');
    }
}

window.__voice_streaming__ = { startStreamingASR, stopStreamingASR, toggleStreaming };  // onclick() event for button with ID recordBtn in chat.html!
// see detailed explainer note on the above pattern at the bottom of this file!



// *****************************************************Recording Functions****************************************************

/*
These functions are part of the simple recording functionality that formed the first implementation of ASR in Privion!

In this implementation, we're using the Web Audio API to record the audio and then upload it to the regular Waitress WSGI server for transcription,
via the `transcribe` API, a regular POST route. This means the user decided when and what to record, whereas with the new ASGI server implementation,
thanks to an async WebSocket, a real-time stream is now established and Privion is in a sense constantly listenning, and knows when to respond thanks to the wake word!

While start-Recording() and stop-Recording() are reused by the new streaming, nonetheless preserving all the recording code as it is functional, useful code and good to have for reference!
*/


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


function writeString(view, offset, string) {
    /*
    Helper function to write a string to a DataView object at a given offset.
    */
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}


function encodeWavPCM16(samplesFloat32, sampleRate) {
    /*
    Function creates a WAV file blob from Float32 audio samples:
    
        - Converts Float32 samples [-1,1] to 16-bit PCM format
        - Constructs a proper WAV file header with RIFF/WAVE format
        - Returns a Blob object that can be used for file downloads or uploads
        
    The WAV format consists of:
        1. RIFF header (12 bytes) - identifies file as RIFF/WAVE
        2. fmt chunk (24 bytes) - describes audio format (PCM, sample rate, etc.)
        3. data chunk header (8 bytes) - identifies start of audio data
        4. PCM samples (variable length) - the actual audio data
    */
    
    // Convert float32 [-1,1] to int16
    const samples = new Int16Array(samplesFloat32.length);
    for (let i = 0; i < samplesFloat32.length; i++) {
        let s = Math.max(-1, Math.min(1, samplesFloat32[i]));
        samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    } // math explained in the float32-ToPCM16 function at the top of the file!

    // Create buffer: 44 bytes for WAV header + 2 bytes per sample (16-bit PCM)
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);    // DataView is a typed array view of an ArrayBuffer, allows for efficient manipulation of the binary data.
    // a typed array is a array with a specific data type, in this case Int16Array, Float32Array, etc.

    // RIFF header (12 bytes total)
    writeString(view, 0, 'RIFF');         // Bytes 0-3: "RIFF" magic number identifies this as a RIFF file
    view.setUint32(4, 36 + samples.length * 2, true);  // Bytes 4-7: File size minus 8 bytes (entire file size - RIFF header)
    writeString(view, 8, 'WAVE');         // Bytes 8-11: "WAVE" format identifier

    // fmt chunk (24 bytes total)
    writeString(view, 12, 'fmt ');        // Bytes 12-15: "fmt " chunk identifier (note the space! It's there to separate the chunk identifier from the chunk data!)
    view.setUint32(16, 16, true);         // Bytes 16-19: Subchunk1Size (16 for PCM format)
    view.setUint16(20, 1, true);          // Bytes 20-21: AudioFormat (1 = uncompressed PCM)
    view.setUint16(22, 1, true);          // Bytes 22-23: NumChannels (1 = mono, 2 = stereo)
    view.setUint32(24, sampleRate, true); // Bytes 24-27: SampleRate (samples per second)
    view.setUint32(28, sampleRate * 2, true); // Bytes 28-31: ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    view.setUint16(32, 2, true);          // Bytes 32-33: BlockAlign (NumChannels * BitsPerSample/8) - bytes per sample frame
    view.setUint16(34, 16, true);         // Bytes 34-35: BitsPerSample (16-bit PCM)

    // data chunk header (8 bytes)
    writeString(view, 36, 'data');       // Bytes 36-39: "data" chunk identifier
    view.setUint32(40, samples.length * 2, true);  // Bytes 40-43: Size of audio data in bytes

    // PCM samples (variable length)
    let offset = 44;  // Start writing audio data after 44-byte header
    for (let i = 0; i < samples.length; i++, offset += 2) {
        view.setInt16(offset, samples[i], true);  // Write each 16-bit sample (little-endian)
    }

    return new Blob([view], { type: 'audio/wav' });
}


function resampleFloat32(input, fromRate, toRate) {
    /*
    Resamples a Float32Array audio buffer from one sample rate to another using linear interpolation:

        - linear interpolation is a simple way to resample audio, it's a weighted average of the two adjacent samples.
        - If the source and target sample rates are the same, returns the input unchanged
        - Calculates the resampling ratio (target_rate / source_rate) to determine output length
        - Uses linear interpolation between adjacent samples to generate intermediate values
        - Handles edge cases where interpolation would go beyond array bounds

    Linear interpolation formula: output = s0 + (s1 - s0) * fraction
    Where:
        - s0 = sample at floor position
        - s1 = sample at ceiling position  
        - fraction = decimal part of the interpolated position

    Example: Resampling from 48kHz to 16kHz (ratio = 16000/48000 = 0.333...)
        - Input has 48000 samples per second
        - Output needs 16000 samples per second
        - Each output sample represents ~3 input samples (1/0.333 = 3)
    */
    
    if (fromRate === toRate) return input;  // No resampling needed
    
    const ratio = toRate / fromRate;  // Resampling ratio: >1 means upsampling, <1 means downsampling
    const newLength = Math.round(input.length * ratio);  // Calculate output array length
    const output = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        // Calculate the corresponding position in the input array for this output sample
        const inPos = i / ratio;  // Map output index back to input space
        
        // Get the integer part (floor) and fractional part for interpolation
        const idx = Math.floor(inPos);  // Index of the sample before the interpolation point
        const frac = inPos - idx;       // Fractional part (0.0 to 0.999...) for interpolation weight
        
        // Get the two samples to interpolate between
        const s0 = input[idx] || 0;           // Current sample (or 0 if beyond array bounds)
        const s1 = input[idx + 1] || s0;      // Next sample (or repeat s0 if at array end)
        
        // Linear interpolation: blend s0 and s1 based on fractional position
        output[i] = s0 + (s1 - s0) * frac;   // When frac=0: output=s0, when frac=1: output=s1
    }
    
    return output;
}


function mergeFloat32(buffers) {
    /*
    Merges multiple Float32Array audio buffers into a single contiguous Float32Array:

        - Takes an array of Float32Array objects (audio chunks) and concatenates them sequentially
        - Calculates the total length needed by summing all individual buffer lengths
        - Creates a new Float32Array with the calculated total length
        - Copies each buffer into the output array at the correct offset position
        - Returns the merged buffer ready for further processing (resampling, encoding, etc.)

    This is commonly used to combine audio chunks collected during recording into a single
    continuous audio stream before processing or encoding.

    Example: If buffers = [Float32Array([1,2]), Float32Array([3,4,5]), Float32Array([6])]
        - Total length = 2 + 3 + 1 = 6
        - Output = Float32Array([1,2,3,4,5,6])
    */
    
    // Calculate total length by summing lengths of all input buffers
    const length = buffers.reduce((acc, source_data) => acc + source_data.length, 0);   // reduce() accumulates: acc starts at 0, adds each buffer's length
    
    // Create output array with exact size needed for all merged data
    const out = new Float32Array(length);
    
    // Track current position in output array for copying each buffer
    let destination_index_offset = 0;
    
    // Copy each buffer sequentially into the output array
    for (const source_data of buffers) {
        out.set(source_data, destination_index_offset);
        
        // Move offset forward by the length of the buffer we just copied
        // This ensures next buffer gets copied to the correct position
        destination_index_offset += source_data.length;
    }
    
    return out;
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

    // Commenting out the below as it's part of the old recording functionality, and we're now using the new streaming functionality
    //--------------------------------
    // // Merge Float32 chunks
    // const merged = mergeFloat32(chunks);
    // // Resample to 16kHz
    // const resampled = resampleFloat32(merged, inputSampleRate, 16000);
    // // Encode WAV (PCM16, mono, 16kHz)
    // const wavBlob = encodeWavPCM16(resampled, 16000);

    // // Send to backend
    // uploadWav(wavBlob);
    //--------------------------------
}


async function startRecording() {
    if (recording) return;
    recording = true;
    chunks = [];

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });  // mouse-hover for additional details! 
        mediaStream = stream;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        inputSampleRate = audioContext.sampleRate;

        sourceNode = audioContext.createMediaStreamSource(stream);

        // Try modern AudioWorklet; fall back to ScriptProcessor only if unavailable
        try {
            // Adjust path if your static URL prefix differs
            await audioContext.audioWorklet.addModule('/static/js/recorder-worklet.js');
            workletNode = new AudioWorkletNode(audioContext, 'recorder-processor');
            /* 
            The AudioWorkletNode interface of the Web Audio API has an associated AudioWorkletProcessor 
            (defined and registered here in the `recorder-worklet.js` file - see detailed explainer note there!), 
            which does the actual audio processing in a Web Audio rendering thread (audio thread).
            */

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
/*
    NOTES ON ABOVE:

    - Known as the "module pattern" or "namespace pattern", this line of code is used to create a global object that contains the start-Recording, stopRecording, & toggleRecord functions.
    - Encapsulation: Instead of polluting the global namespace with multiple individual functions, it groups related functionality under one namespace object
    - Commonly used JS pattern to organize and manage complex functionality, providing a clean API surface for external code & making debugging easier
    - Selectively exposes those functions that form part of the complete public API for voice streaming functionality, and thus need to be accessible from outside this module (particularly from HTML elements).
    
    Given the above, we're purely using this pattern for clarity:
    While all functions from voices.js  would be imported into the global namespace as we do the same import `<script src="{{ url_for('static', filename='js/voices.js') }}"></script>` in chat.html,
    this module contains so many related helper functions that explicitly specifying control functions for external use is helpful for clarity & maintainability.

    Also if this file becomes a proper ES module, the functions won't be automatically exported to the global namespace, so this pattern is necessary to ensure they're accessible!
    
    Lastly, do note that `{ startStreamingASR, stopStreamingASR, toggleStreaming }` syntax is ES6 object literal shorthand:
    it's equivalent to `{ startStreamingASR: startStreamingASR, stopStreamingASR: stopStreamingASR, toggleStreaming: toggleStreaming }`
*/