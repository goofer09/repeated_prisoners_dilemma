from payoff import C,D
from policies import AlwaysC,AlwaysD,TitForTat
from environment import RepeatedPrisonersDilemmaEnv


def summarize_trajectory(trajectory):
    T=len(trajectory)
    if T==0:
        return {"T":0,"total_r1":0.0,"total_r2":0.0,"cc_rate":0.0,"cooperation_rate":0.0}
    
    total_r1=sum(row["r1"]for row in trajectory)
    total_r2=sum(row["r2"]for row in trajectory)

    cc=0
    coop_actions=0

    for row in trajectory:
        a1,a2=row["a1"],row["a2"]
        if (a1,a2)==(C,C):
            cc+=1
        if a1==C:
            coop_actions+=1
        if a2==C:
            coop_actions+=1

    return {
        "T":T,
        "total_r1": total_r1,
        "total_r2": total_r2,
        "avg_r1": total_r1 / T,
        "avg_r2": total_r2 / T,
        "cc_count": cc,
        "cc_rate": cc / T,
        "cooperation_rate": coop_actions / (2 * T),
    }
        

def rollout_episode(env,policy1,policy2,horizon):

    trajectory=[]
    state=env.reset()
    done=False

    for t in range(horizon):
        if done:
            break
        
        a1_intended=policy1.act(state,player_id=1)
        a2_intended=policy2.act(state,player_id=2)
        
        flip1,flip2=False,False ##deterministic approach for now
        a1=a1_intended
        a2=a2_intended

        next_state,(r1,r2),done=env.step(a1,a2) ##assigment from function output

        trajectory.append({
            "t":t,
            "state":state,
            "a1_intended": a1_intended,
            "a2_intended": a2_intended,
            "a1": a1,
            "a2": a2,
            "r1": r1,
            "r2": r2,
            "next_state": next_state,
            "flip1": flip1,
            "flip2": flip2,
        })

        state = next_state

    summary=summarize_trajectory(trajectory)
    return trajectory,summary


##Example Usage
if __name__ == "__main__":
    traj, summary = rollout_episode(RepeatedPrisonersDilemmaEnv(), TitForTat(), AlwaysD(), horizon=1000000)

    print("SUMMARY:", summary)
    print("FIRST 2 STEPS:")
    for row in traj[:2]:
        print(row)

