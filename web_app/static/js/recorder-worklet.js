/*
Runs in AudioWorkletGlobalScope
--------------------------------
The AudioWorkletProcessor runs on the Web Audio rendering thread (separate from main JavaScript thread)
for high-performance, low-latency audio processing without blocking the UI. The corresponding 
AudioWorkletNode runs on the main thread and communicates via MessagePort for data transfer.
--------------------------------
*/

class RecorderProcessor extends AudioWorkletProcessor {
    /**
     * Process audio data in real-time from microphone input.
     * 
     * This method is called automatically by the Web Audio API for each audio quantum (128 samples).
     * The 128-sample quantum size is a Web Audio API standard that balances latency and performance:
     * - At 48kHz sample rate: 128 samples = ~2.67ms of audio per quantum
     * - At 44.1kHz sample rate: 128 samples = ~2.9ms of audio per quantum
     * - This provides low-latency processing while maintaining efficient CPU usage
     * 
     * It captures mono audio data from the microphone, transfers it to the main thread for processing,
     * and outputs silence to prevent audio feedback while maintaining audio graph connectivity.
     * 
     * @param {Float32Array[][]} inputs - Array of input audio channels. inputs[0] is the first input node,
     *                                   inputs[0][0] is the first channel (left/mono) of that input.
     *                                   Each channel contains 128 Float32 samples in range [-1.0, 1.0].
     * @param {Float32Array[][]} outputs - Array of output audio channels. outputs[0] is the first output node,
     *                                    outputs[0][0] is the first channel where we write processed audio.
     *                                    We fill this with zeros to prevent microphone feedback.
     * @returns {boolean} - True to keep the processor alive, false to terminate it.
     */
    process(inputs, outputs) {
        const input = inputs[0];  // Get the first (and typically only) input node
        if (input && input[0]) {
            const ch = input[0]; // Extract mono channel (first channel) - Float32Array of 128 samples
            
            // Copy audio data to a transferable buffer for zero-copy transfer to main thread
            // We create a new Float32Array because the original 'ch' buffer is owned by the Web Audio API
            // and will be reused for the next audio quantum. Without copying, we'd lose the data.
            const copy = new Float32Array(ch.length);  // Create new buffer with same length (128 samples)
            copy.set(ch);  // Copy all samples from source to destination: copy[i] = ch[i] for all i
            
            // Transfer audio data to main thread with ownership transfer for optimal performance
            // - copy.buffer is the underlying ArrayBuffer containing the Float32 sample data
            // - [copy.buffer] is the transfer list, making the copy buffer unavailable here in the worklet thread
            // - Instead, the main thread now owns this copy & receives it via AudioWorkletNode's port.onmessage event handler
            // - This transfer mechanism avoids expensive memory copying, providing zero-copy data transfer
            this.port.postMessage(copy.buffer, [copy.buffer]);
        }

        // Output silence to prevent audio feedback while maintaining audio graph connectivity
        // Even though we're capturing input, Web Audio API requires nodes to have outputs to stay connected
        const output = outputs[0];  // Get the first (and typically only) output node
        if (output && output[0]) {
            // Fill output buffer with zeros (silence) to prevent microphone audio from reaching speakers
            // - output[0] is the first channel (left/mono) of the output
            // - fill(0) sets all 128 samples to 0.0, creating digital silence
            // - This prevents audio feedback loops while keeping the node active in the audio graph
            output[0].fill(0);
        }
        
        return true; // Return true to keep the processor alive and continue processing audio quanta
    }
}

// Register our custom processor with the AudioWorklet system
// The string 'recorder-processor' is the identifier used when creating the AudioWorkletNode in the main thread
registerProcessor('recorder-processor', RecorderProcessor);