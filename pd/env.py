from payoff import get_payoff, C, D

class RepeatedPrisonersDilemmaEnv:
    def __init__(self, horizon=None):
        self.horizon = horizon
        self.current_step = 0
        self.state = None  # State can be defined as needed, e.g., history of actions

    def reset(self):
        self.current_step=0
        self.state = None  # Reset state as needed
        return self.state
    
    def step(self, action1, action2):
        if self.horizon is not None and self.current_step > self.horizon:
            raise ValueError("Horizon reached. Please reset the environment.")
        
        prev_state = self.state
        print("PREV_STATE:", prev_state)
        payoff_1,payoff_2 = get_payoff(action1,action2)
        print("ACTIONS:", action1, action2)
        print("PAYOFFS:", payoff_1, payoff_2)


        next_state = (action1, action2)  # Example state representation, can be modified as needed
        self.state = next_state
        self.current_step += 1
        done = self.horizon is not None and self.current_step == self.horizon
        print("STORED_STATE:", self.state)
        print("NEXT_STATE:", next_state)
        print("STEP:", self.current_step, "DONE:", done)

        return next_state, (payoff_1, payoff_2), done

    
###/
# Example usage
if __name__ == "__main__":
    env = RepeatedPrisonersDilemmaEnv(horizon=5)
    state = env.reset()
    print(state)  # Initial state
    next_state, payoffs, done = env.step(C, C)
    print(next_state)  # Next state after actions
    print(payoffs)  # Payoffs for both players
    print(done)  # Whether the episode is done
    ####
