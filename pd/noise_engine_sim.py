import numpy as np
from engine_copy import rollout_episode
from environment import RepeatedPrisonersDilemmaEnv
from policies import TitForTat, AlwaysD

rng = np.random.default_rng(123)
traj, summary = rollout_episode(
    RepeatedPrisonersDilemmaEnv(),
    TitForTat(),
    TitForTat(),
    horizon=10000,
    epsilon=0.2,
    rng=rng,
)

flip1_rate = sum(int(r["flip1"]) for r in traj) / len(traj)
flip2_rate = sum(int(r["flip2"]) for r in traj) / len(traj)
both_rate = sum(int(r["flip1"] and r["flip2"]) for r in traj) / len(traj)

print("summary:", summary)
print("flip1_rate:", flip1_rate)
print("flip2_rate:", flip2_rate)
print("both_rate:", both_rate, " expected~", flip1_rate * flip2_rate)
