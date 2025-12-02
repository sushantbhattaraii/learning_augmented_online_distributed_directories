import random

# --- Setup Data (Example) ---
# Notice we have duplicates in Vp (1000 appears twice)
Vp_main = [1000, 2000, 1000, 3000, 4000, 2000, 5000, 6000]
Q_main  = [0.5,  0.7,  0.9,  1.2,  1.5,  0.8,  1.1,  1.3]
VpAndQ = list(zip(Vp_main, Q_main))

# --- Configuration ---
# 1. Determine how many to extract (between 1 and half length)
total_len = len(VpAndQ)
max_extract = max(1, total_len // 2) # Ensure at least 1
num_to_extract = random.randint(1, max_extract)

# 2. Create a working copy of the pool so we can remove items as we pick them
pool = list(VpAndQ)
selected_pairs = []

print(f"Attempting to extract {num_to_extract} pairs...")

# --- Selection Logic ---
for _ in range(num_to_extract):
    if not selected_pairs:
        # First selection: All items in pool are valid candidates
        candidates = pool
    else:
        # Subsequent selections: Filter candidates
        # Rule: candidate's Vp must not equal the last selected Vp
        last_vp = selected_pairs[-1][0]
        candidates = [pair for pair in pool if pair[0] != last_vp]

    # Safety Check: If we run out of valid candidates (corner case)
    if not candidates:
        print("Warning: Ran out of valid non-consecutive options early.")
        break

    # 3. Pick a random valid candidate
    choice = random.choice(candidates)
    
    # 4. Add to result and remove from the available pool
    selected_pairs.append(choice)
    pool.remove(choice)

# --- Output ---
print("Selected Sequence:")
print(selected_pairs)