// Runs on the audio rendering thread. Buffers incoming Float32 audio blocks, converts to
// Int16 PCM once enough is buffered (~250ms), and posts the raw bytes to the main thread.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunks = [];
    this._bufferedSamples = 0;
    this._chunkSamples = 4000; // ~250ms at 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0]; // mono
      this._chunks.push(channelData.slice());
      this._bufferedSamples += channelData.length;

      if (this._bufferedSamples >= this._chunkSamples) {
        const merged = new Float32Array(this._bufferedSamples);
        let offset = 0;
        for (const chunk of this._chunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }

        const int16 = new Int16Array(merged.length);
        for (let i = 0; i < merged.length; i++) {
          const s = Math.max(-1, Math.min(1, merged[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        this.port.postMessage(int16.buffer, [int16.buffer]);
        this._chunks = [];
        this._bufferedSamples = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorkletProcessor);
