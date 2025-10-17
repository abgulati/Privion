// Runs in AudioWorkletGlobalScope
class RecorderProcessor extends AudioWorkletProcessor {
    process(inputs, outputs) {
        const input = inputs[0];
        if (input && input[0]) {
            const ch = input[0]; // mono
            // Copy to a transferable buffer
            const copy = new Float32Array(ch.length);
            copy.set(ch);
            this.port.postMessage(copy.buffer, [copy.buffer]);
        }

        // Output silence to keep the node connected without echo
        const output = outputs[0];
        if (output && output[0]) {
            output[0].fill(0);
        }
        return true; // keep alive
    }
}
registerProcessor('recorder-processor', RecorderProcessor);