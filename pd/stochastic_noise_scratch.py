from payoff import C, D
import numpy as np
from typing import Tuple

def flip_action(a):
    if a not in (C, D):
        raise ValueError(f"Invalid action: {a!r}. Expected {C} (C) or {D} (D).")
    if a == C:
        return D
    return C

def sample_flip(epsilon:float,rng:np.random.Generator)->bool:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
    return bool(rng.random() < epsilon)


##if __name__ == "__main__":
    ##rng = np.random.default_rng(42)
    ##print(sample_flip(0.9, rng))


def apply_noise(a: int, epsilon: float, rng: np.random.Generator) -> Tuple[int, bool]:
    flipped = sample_flip(epsilon, rng)
    realized_a = flip_action(a) if flipped else a
    return realized_a, flipped


if __name__ == "__main__":
    # Keep one RNG instance and reuse it for all draws.
    rng = np.random.default_rng(42)

    n = 1000000
    flips = 0
    for _ in range(n):
        _, flipped = apply_noise(C, 0.2124, rng)
        flips += int(flipped)

    print(f"flips={flips} / {n} ({flips / n:.3f})")


def apply_noise_pair(a1:int,a2:int,epsilon: float, rng: np.random.Generator) -> Tuple[int, int, bool,bool]:
    flip_1=sample_flip(epsilon,rng)
    flip_2=sample_flip(epsilon,rng)
    realized_a1=flip_action(a1) if flip_1 else a1
    realized_a2=flip_action(a2) if flip_2 else a2

    return realized_a1,realized_a2,flip_1,flip_2

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    epsilon = 0.30
    n = 1_000_000

    flips_1 = 0
    flips_2 = 0
    joint_counts = {
        (False, False): 0,
        (False, True): 0,
        (True, False): 0,
        (True, True): 0,
    }

    for _ in range(n):
        _, _, flip_1, flip_2 = apply_noise_pair(C, D, epsilon, rng)
        flips_1 += int(flip_1)
        flips_2 += int(flip_2)
        joint_counts[(flip_1, flip_2)] += 1

    p1 = flips_1 / n
    p2 = flips_2 / n
    p11 = joint_counts[(True, True)] / n
    indep_gap = p11 - (p1 * p2)

    print(f"epsilon={epsilon}, n={n}")
    print(f"player1 flip rate: {p1:.4f}")
    print(f"player2 flip rate: {p2:.4f}")
    print("joint flip probabilities:")
    print(f"  P(F1=0,F2=0) = {joint_counts[(False, False)] / n:.4f}")
    print(f"  P(F1=0,F2=1) = {joint_counts[(False, True)] / n:.4f}")
    print(f"  P(F1=1,F2=0) = {joint_counts[(True, False)] / n:.4f}")
    print(f"  P(F1=1,F2=1) = {p11:.4f}")
    print(f"independence check: P11 - P1*P2 = {indep_gap:.6f}")

    print("\nDeterministic edge checks for two-player noise:")
    r1, r2, f1, f2 = apply_noise_pair(C, D, 0.0, rng)
    print(f"epsilon=0.0 -> realized=({r1},{r2}), flipped=({f1},{f2})")

    r1, r2, f1, f2 = apply_noise_pair(C, D, 1.0, rng)
    print(f"epsilon=1.0 -> realized=({r1},{r2}), flipped=({f1},{f2})")
