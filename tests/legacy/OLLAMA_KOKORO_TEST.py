import time
import sounddevice as sd
from kokoro import KPipeline

# ==========================================================
# CONFIG
# ==========================================================

TEXT = """
You are standing in an office.

There is a chair directly ahead.

The path is clear on your left.

A door is approximately three meters away.

Please proceed carefully.
"""

VOICE = "af_heart"

# ==========================================================
# LOAD
# ==========================================================

print("=" * 70)
print("LOADING KOKORO")
print("=" * 70)

t0 = time.perf_counter()

pipeline = KPipeline(lang_code="a")

t1 = time.perf_counter()

print(f"Pipeline Loaded : {(t1-t0):.2f} sec")

print()

input("Press ENTER to synthesize...")

# ==========================================================
# SYNTHESIS
# ==========================================================

print()
print("Generating...")
print()

t_generate = time.perf_counter()

generator = pipeline(
    TEXT,
    voice=VOICE,
    speed=1.0,
)

chunks = []

total_samples = 0

for _, _, audio in generator:

    chunks.append(audio)

    total_samples += len(audio)

t_generated = time.perf_counter()

print("Generation Finished")

print()

print(f"Chunks : {len(chunks)}")
print(f"Samples: {total_samples}")

# ==========================================================
# PLAYBACK
# ==========================================================

print()
print("Playing...")
print()

t_play = time.perf_counter()

for audio in chunks:

    sd.play(audio, 24000)
    sd.wait()

t_end = time.perf_counter()

# ==========================================================
# RESULTS
# ==========================================================

print()

print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Pipeline Load : {(t1-t0):.2f} sec")

print(f"Synthesis     : {(t_generated-t_generate):.2f} sec")

print(f"Playback      : {(t_end-t_play):.2f} sec")

print()

print(f"TOTAL         : {(t_end-t_generate):.2f} sec")

print("=" * 70)