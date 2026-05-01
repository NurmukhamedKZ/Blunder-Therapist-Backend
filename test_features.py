"""Quick smoke test: does feature extraction work on a real game?"""
import sys
sys.path.insert(0, "/home/claude/blunder-therapist/backend")

from app.services.features import extract_features, features_to_llm_summary

# A fictional 12-move game with a deliberate blunder cascade by black
# Scholar's mate setup, then a blunder cascade
PGN = """
[Event "Test"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

# Synthetic eval per ply (centipawns from White's POV)
# A move with big swing = blunder
eval_per_ply = [
    20,    # 1. e4 - balanced
    0,     # 1...e5 - balanced
    30,    # 2. Bc4
    25,    # 2...Nc6
    -50,   # 3. Qh5 - dubious but not lost
    400,   # 3...Nf6?? - massive swing toward white = BLACK BLUNDER (Nf6 missed Qxf7#)
    9999,  # 4. Qxf7# - mate
]

# Per-ply thinking time (seconds)
# Black panic-rushed move 4 (the blunder) at 1.0s after thinking 8s normally
time_per_ply = [
    5.0,   # white thought 5s
    4.0,   # black 4s
    3.0,
    8.0,
    6.0,
    1.0,   # black RUSHED on the critical move - should trigger tilt signal
    2.0,
]

features = extract_features(
    pgn=PGN,
    eval_per_ply=eval_per_ply,
    time_per_ply=time_per_ply,
    player_color="black",
    result="loss",
)

print("=== EXTRACTED FEATURES ===")
print(f"Blunders: {features.blunder_count}")
print(f"Avg time overall: {features.avg_time_overall}s")
print(f"Avg time on blunders: {features.avg_time_before_blunders}s")
print(f"Blunder-speed ratio: {features.blunder_speed_ratio}")
print(f"Cascade detected: {features.has_blunder_cascade}")
print()
print("=== LLM-READY SUMMARY ===")
print(features_to_llm_summary(features))
