# main.py
# Real-time pipeline: captures mic input, transcribes, translates, and speaks back


import os
import queue
import threading
import time
import argparse
from pydub import AudioSegment
from pydub.playback import play
import logging


from config import RATE, CHUNK, CHANNELS
from audio_utils import write_wav_file, audio_interface as p
from asr import load_asr_model, transcribe
from translate import load_translation_model, translate_text
from tts import load_tts, synthesize_speech


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="auto", help="Source language code")
parser.add_argument("--tgt", required=True, help="Target language code")
args = parser.parse_args()

FORMAT = p.get_format_from_width(2)
CHUNK_DURATION = 3  # seconds
MIC_INDEX = 1       # Realtek mic

# Records audio from the microphone in fixed-size chunks and pushes them to a queue
def record(q, stop):
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=MIC_INDEX,
        frames_per_buffer=CHUNK
    )
    print(">>> Recording… (buffering 3s chunks — Ctrl+C to stop)")
    while not stop.is_set():
        frames = [
            stream.read(CHUNK)
            for _ in range(int(RATE / CHUNK * CHUNK_DURATION))
        ]
        q.put(b''.join(frames))
        print("[recorded chunk]")
    stream.stop_stream()
    stream.close()

# Consumes audio chunks, performs ASR, translation, and speech synthesis, and plays the result
def process(q_in, stop):
    asr_model = load_asr_model()
    trans_model, trans_tok = load_translation_model()
    tts_engine = load_tts(args.tgt)

    while not stop.is_set():
        if not q_in.empty():
            data = q_in.get()
            in_wav, out_wav = "temp_in.wav", "temp_out.wav"
            write_wav_file(in_wav, data)

            try:
                text = transcribe(in_wav, asr_model, args.src)
                logger.debug("Raw transcription output: %s", text)
                print(f"[DEBUG] Raw transcription output: {text}")
                if text:
                    print(f"Transcribed: {text}")
                    translated = translate_text(text, args.src, args.tgt, trans_model, trans_tok)
                    print(f"Translated: {translated}")
                    synthesize_speech(translated, tts_engine, out_wav)
                    play(AudioSegment.from_wav(out_wav))
                    os.remove(out_wav)
                else:
                    print("[DEBUG] No transcription result.")
            except Exception as e:
                print(f"[ERROR] Transcription or translation failed: {e}")

            os.remove(in_wav)

# Starts recording and processing threads; runs until manually interrupted
def main():
    audio_q = queue.Queue()
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=record, args=(audio_q, stop_event), daemon=True),
        threading.Thread(target=process, args=(audio_q, stop_event), daemon=True)
    ]

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        p.terminate()
        print("Exiting...")

if __name__ == "__main__":
    main()