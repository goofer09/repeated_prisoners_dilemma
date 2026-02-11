from environment import RepeatedPrisonersDilemmaEnv
from engine import rollout_episode
from payoff import C, D
from policies import AlwaysC, AlwaysD, TitForTat

#1)Spec Registry
#Policy Runs and horizon
# claims to assert later

BEHAVIORAL_SPEC={
    "TfTvsAC":{
        "Policy1":TitForTat,
        "Policy2":AlwaysC,
        "horizon":10,
        "exp_behavior":{
            "expected_prefix":(C,C),
            "constant_joint_action":(C,C),
            "burn_in_max":1,
            "tail_joint_actions":(C,C),
            "tail_len":5,
        }
    },
    "TfTvsAD":{
        "Policy1":TitForTat,
        "Policy2":AlwaysD,
        "horizon":10,
        "exp_behavior":{
            "expected_prefix":(C,D),
            "constant_joint_action":(D,D),
            "burn_in_max":1,
            "tail_joint_actions":(D,D),
            "tail_len":5,
        }
    },
    "TfTvsTfT":{
        "Policy1":TitForTat,
        "Policy2":TitForTat,
        "horizon":10,
        "exp_behavior":{
            "expected_prefix":(C,C),
            "constant_joint_action":(C,C),
            "burn_in_max":1,
            "tail_joint_actions":(C,C),
            "tail_len":5,
        }
    },

}

##2) Case Resolver

def resolve_behavior_case(TestCase_key):
    TestCase=BEHAVIORAL_SPEC[TestCase_key]
    policy1=TestCase["Policy1"]()
    policy2=TestCase["Policy2"]()
    horizon=TestCase["horizon"]
    expected_behavior=TestCase["exp_behavior"]

    return policy1,policy2,horizon,expected_behavior


def analyze_behavior(trajectory,summary):
    action_pairs = [(row["a1"], row["a2"]) for row in trajectory]

    #State consistency checks

    state_ok=True
    state_fail_at=None
    for t, row in enumerate(trajectory):
        if row["next_state"]!=(row["a1"],row["a2"]):##failure of action update to state
            state_ok=False
            state_fail_at=t
            break
        if t>0 and trajectory[t]["state"]!=trajectory[t-1]["next_state"]:##failure of state transition
            state_ok=False
            state_fail_at=t
            break
    
    return {
        "action_pairs":action_pairs,
        "summary":summary,
        "same_intended_realized":all(
            r["a1_intended"]==r["a1"] and r["a2_intended"]==r["a2"]
            for r in trajectory
        ),
        "state_transition_ok":state_ok,
        "state_transition_fail_at":state_fail_at,
        "head10":action_pairs[:10],
        "tail10":action_pairs[-10:],
    }

##4)Assertion Principles

def assert_prefix(action_pairs,expected_prefix):
    if not action_pairs:
        raise AssertionError("no action observed, expected atleast one")
    if action_pairs[0]!=expected_prefix:
        raise AssertionError(f"first action mismatch:observed={action_pairs[0]},expected={expected_prefix}" )
    
    assert action_pairs[0]==expected_prefix
    pass

def assert_eventually_constant(action_pairs,constant_joint_action,burn_in_max):
    if not action_pairs:
        raise AssertionError("no actions observed")
    last_k=min(burn_in_max,len(action_pairs)-1)

    for k in range (last_k+1):
        if all(a==constant_joint_action for a in action_pairs[k:]):
            return
        
    raise AssertionError(f"not eventually constant as {constant_joint_action} within burn_in_max={burn_in_max}")

def assert_tail_all(action_pairs, constant_joint_action, tail_len):
    if tail_len < 0:
        raise AssertionError(f"tail_len must be >= 0, got {tail_len}")
    if tail_len == 0:
        return
    if len(action_pairs) < tail_len:
        raise AssertionError(
            f"not enough actions: need tail_len={tail_len}, observed={len(action_pairs)}"
        )

    tail = action_pairs[-tail_len:]
    for i, observed in enumerate(tail):
        if observed != constant_joint_action:
            raise AssertionError(
                f"tail mismatch at tail index {i}: "
                f"observed={observed}, expected={constant_joint_action}"


def assert_alternating(joint_actions, pattern, start_after):
    if not pattern:
        raise AssertionError("pattern must be non-empty")
    if start_after < 0:
        raise AssertionError(f"start_after must be >= 0, got {start_after}")
    if start_after >= len(joint_actions):
        raise AssertionError(
            f"start_after={start_after} out of range for {len(joint_actions)} actions"
        )

    m = len(pattern)
    for t in range(start_after, len(joint_actions)):
        expected = pattern[(t - start_after) % m]
        observed = joint_actions[t]
        if observed != expected:
            raise AssertionError(
                f"alternating mismatch at index {t}: observed={observed}, expected={expected}"
            )


##5) Claim dispatcher

def validate_behavior_claims(derived,claims):

    realized_action_pairs=derived[action_pairs]

    for behavior in exp_behavior:
    if "expected_prefix" in behavior:
        assert_prefix(action_pairs, behavior["expected_prefix"])
    elif "constant_joint_action" in behavior:
        cfg = behavior["constant_joint_action"]
        assert_eventually_constant(joint_actions, cfg["joint_action"], cfg["burn_in_max"])
    elif "tail_all" in behavior:
        cfg = behavior["tail_all"]
        assert_tail_all(joint_actions, cfg["joint_action"], cfg["tail_len"])
    elif "alternating" in behavior:
        cfg = behavior["alternating"]
        assert_alternating(action_pairs, cfg["pattern"], cfg["start_after"])
    else:
        raise ValueError("Unknown claim format")







