import math

def entropy_node(y):
    n = len(y)
    counts = {}

    # count occurrences of each class
    for label in y:
        counts[label] = counts.get(label, 0) + 1

    H = 0.0

    for count in counts.values():
        p = count / n
        if p > 0:
            H -= p * math.log2(p)

    return H