import re

# Parse the comprehensive benchmark data
data = """
"""

# Parse the data
lines = [line.strip() for line in data.strip().split(
    '\n') if '[fcl]' in line or '[bonsai]' in line]

fcl_tree = []
fcl_collision = []
bonsai_tree = []
bonsai_collision = []

for line in lines:
    if '[fcl]    tree construction' in line:
        match = re.search(r': (\d+) ms', line)
        if match:
            fcl_tree.append(int(match.group(1)))
    elif '[fcl]    collision detection' in line:
        match = re.search(r': (\d+) ms', line)
        if match:
            fcl_collision.append(int(match.group(1)))
    elif '[bonsai] tree construction' in line:
        match = re.search(r': (\d+) ms', line)
        if match:
            bonsai_tree.append(int(match.group(1)))
    elif '[bonsai] collision detection' in line:
        match = re.search(r': (\d+) ms', line)
        if match:
            bonsai_collision.append(int(match.group(1)))

print("Total measurements:")
print(f"FCL tree construction: {len(fcl_tree)}")
print(f"FCL collision detection: {len(fcl_collision)}")
print(f"Bonsai tree construction: {len(bonsai_tree)}")
print(f"Bonsai collision detection: {len(bonsai_collision)}")

# Calculate averages
avg_fcl_tree = sum(fcl_tree) / len(fcl_tree)
avg_fcl_collision = sum(fcl_collision) / len(fcl_collision)
avg_bonsai_tree = sum(bonsai_tree) / len(bonsai_tree)
avg_bonsai_collision = sum(bonsai_collision) / len(bonsai_collision)

print(f"\nAverages:")
print(f"FCL tree construction: {avg_fcl_tree:.1f} ms")
print(f"FCL collision detection: {avg_fcl_collision:.1f} ms")
print(f"Bonsai tree construction: {avg_bonsai_tree:.1f} ms")
print(f"Bonsai collision detection: {avg_bonsai_collision:.1f} ms")

# Calculate ratios
tree_ratio = avg_bonsai_tree / avg_fcl_tree
collision_ratio = avg_bonsai_collision / avg_fcl_collision

print(f"\nBonsai vs FCL ratios:")
print(f"Tree construction ratio: {tree_ratio:.2f}")
print(f"Collision detection ratio: {collision_ratio:.2f}")

# Calculate speedup/slowdown
if tree_ratio < 1.0:
    print(f"Bonsai tree construction is {1/tree_ratio:.2f}x faster")
else:
    print(f"Bonsai tree construction is {tree_ratio:.2f}x slower")

if collision_ratio < 1.0:
    print(f"Bonsai collision detection is {1/collision_ratio:.2f}x faster")
else:
    print(f"Bonsai collision detection is {collision_ratio:.2f}x slower")
