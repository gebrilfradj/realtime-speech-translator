# benchmark.py
# Benchmarks end-to-end latency of the speech translation pipeline

import time
import argparse
import torch

from audio_utils import write_wav_file
from asr import load_asr_model, transcribe
from translate import load_translation_model, translate_text
from tts import load_tts, synthesize_speech

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='en')
parser.add_argument('--tgt', default='es')
parser.add_argument('--audio', required=True, help='Path to sample WAV')
parser.add_argument('--repeats', type=int, default=5)
args = parser.parse_args()

# Performs multiple translation cycles and reports average latency
def measure():
    # Load ASR, translation, and TTS models
    asr = load_asr_model()
    t_model, t_tok = load_translation_model()
    tts = load_tts(args.tgt)
    latencies = []
    for _ in range(args.repeats):
        # Begin latency timer
        start = time.time()
        txt = transcribe(args.audio, asr, args.src)
        tr = translate_text(txt, args.src, args.tgt, t_model, t_tok)
        synthesize_speech(tr, tts, 'bench_out.wav')
        latencies.append(time.time() - start)
    avg = sum(latencies) / len(latencies)
    # Compute and print average latency and device info
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    print(f"Avg latency: {avg:.2f}s on {device}")

if __name__ == '__main__':
    measure()