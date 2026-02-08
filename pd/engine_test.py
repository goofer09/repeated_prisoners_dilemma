from environment import RepeatedPrisonersDilemmaEnv
from engine import rollout_episode
from payoff import C, D
hard invariant checks

def ACvAC:
    expected_dict= {
      "expected_joint_action_first":(C,C),
      "expected_joint_action_steady":(C,C),
      "expected_cc_count":5,
      "expected_coop_rate":1.0,
      "expected_symmetry":r1==r2
    }
rollout_episode(RepeatedPrisonersDilemaEnv(),AlwaysC(),AlwaysC(),horizon=5)

class ADvAD

class ACvAD
class invertACAD

behavioural checks

class TFTvAC

class TFTvAD

class TFTvTFT