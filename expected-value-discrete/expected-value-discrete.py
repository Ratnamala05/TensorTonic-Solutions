def expected_value_discrete(x, p):
    if len(x) != len(p):
        raise ValueError("Length mismatch")

    if any(pi < 0 for pi in p):
        raise ValueError("Negative probability")

    if abs(sum(p) - 1.0) > 1e-9:
        raise ValueError("Probabilities must sum to 1")

    ev = 0
    for xi, pi in zip(x, p):
        ev += xi * pi

    return ev