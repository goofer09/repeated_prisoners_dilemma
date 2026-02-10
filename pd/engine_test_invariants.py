from environment import RepeatedPrisonersDilemmaEnv
from engine import rollout_episode
from payoff import C, D
from policies import AlwaysC,AlwaysD,TitForTat

##hard invariant checks

Hard_Spec={
    "ACvsAC":{
        "Policy1":AlwaysC,
        "Policy2":AlwaysC,
        "horizon":5,
        "exp_summary":{
            "first_joint_action":(C,C),
            "steady_from":0,
            "steady_joint_action":(C,C),
            "cc_count":5,
            "coop_rate":1.0,    
        },    
    },
    "ADvsAD":{
        "Policy1":AlwaysD,
        "Policy2":AlwaysD,
        "horizon":5,
        "exp_summary":{
            "first_joint_action":(D,D),
            "steady_from":0,
            "steady_joint_action":(D,D),
            "cc_count":0,
            "coop_rate":0.0,    
        },    
    },
    "ACvsAD":{
        "Policy1":AlwaysC,
        "Policy2":AlwaysD,
        "horizon":5,
        "exp_summary":{
            "first_joint_action":(C,D),
            "steady_from":0,
            "steady_joint_action":(C,D),
            "cc_count":0,
            "coop_rate":0.5,
        }
    }
}

def resolve_test_case(TestCase_key):
    TestCase=Hard_Spec[TestCase_key]
    policy1=TestCase["Policy1"]()
    policy2=TestCase["Policy2"]()
    horizon=TestCase["horizon"]
    expected_summary=TestCase["exp_summary"]

    return policy1,policy2,horizon,expected_summary

##trajectory,summary=rollout_episode(RepeatedPrisonersDilemmaEnv,test_rollout_epsiode())

def analyze_traj(trajectory,summary):
    action_pairs = [(row["a1"], row["a2"]) for row in trajectory]

    state_ok=True
    state_fail_at=None
    for t , row in enumerate(trajectory):
        #local state transition check
        if row["next_state"]!=(row["a1"],row["a2"]):
            state_ok=False
            state_fail_at=t
            break
        #cross state check
        if t>0 and trajectory[t]["state"]!=trajectory[t-1]["next_state"]:
            state_ok=False
            state_fail_at=t
            break
        
    return {
        "action_pairs":action_pairs,
        "state_transition_ok":state_ok,
        "state_transition_fail_at":state_fail_at,
        "same_intended_realized":all(
            r["a1_intended"]==r["a1"] and r["a2_intended"]==r["a2"]
            for r in trajectory
        ),
        "summary":summary,
    }

##Validator: expected claims checked

def validate_TestCase(derived,expected_summary):
   ## first action check 
    if "first_joint_action" in expected_summary:
        assert derived["action_pairs"][0]==expected_summary["first_joint_action"]

    ## steady action check
    if "steady_from" in expected_summary and "steady_joint_action" in expected_summary:
        k=expected_summary["steady_from"]
        target=expected_summary["steady_joint_action"]
        assert all(p==target for p in derived["action_pairs"][k:])

    summary = derived["summary"]
    if "cc_count" in expected_summary:
        assert summary["cc_count"] == expected_summary["cc_count"]
    if "coop_rate" in expected_summary:
        assert abs(summary["cooperation_rate"] - expected_summary["coop_rate"]) < 1e-12
    if "cooperation_rate" in expected_summary:
        assert abs(summary["cooperation_rate"] - expected_summary["cooperation_rate"]) < 1e-12

    ## always on invariants
    assert derived["same_intended_realized"]
    assert derived["state_transition_ok"]


def run_hard_check(selection):
    TestCase_keys=Hard_Spec.keys() if selection =="all" else [selection]

    report={}
    for TestCase_key in TestCase_keys:
        policy1,policy2,horizon,expected_summary=resolve_test_case(TestCase_key)
        trajectory,summary = rollout_episode(RepeatedPrisonersDilemmaEnv(),policy1,policy2,horizon)
        derived=analyze_traj(trajectory,summary)
        validate_TestCase(derived,expected_summary)
        report[TestCase_key]="pass"

    return report

print(run_hard_check("all"))
