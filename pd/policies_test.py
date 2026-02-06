from policies import AlwaysC, AlwaysD, TitForTat
from payoff import C, D

# happy path
assert AlwaysC().act(None, 1) == C
assert AlwaysD().act(None, 2) == D
assert TitForTat().act(None, 1) == C
assert TitForTat().act((C, D), 1) == D
assert TitForTat().act((C, D), 2) == C

# validation path
for bad_player in (0, 3):
    try:
        AlwaysC().act(None, bad_player)
        raise AssertionError("Expected ValueError for bad player_id")
    except ValueError:
        pass

for bad_state in ((C,), (C, 99), (99, D)):
    try:
        TitForTat().act(bad_state, 1)# type: ignore[arg-type]
        raise AssertionError("Expected ValueError for bad state")
    except ValueError:
        pass

print("All policy checks passed")
