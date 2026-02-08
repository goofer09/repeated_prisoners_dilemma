from payoff import get_payoff, C, D

class RepeatedPrisonersDilemmaEnv:
    def __init__(self):
        self.current_step = 0
        self.state = None  # State can be defined as needed, e.g., history of actions

    def reset(self):
        self.state = None  # Reset state as needed
        return self.state
    
    def step(self, action1, action2):
        
        prev_state = self.state
        payoff_1,payoff_2 = get_payoff(action1,action2)


        next_state = (action1, action2)  # Example state representation, can be modified as needed
        self.state = next_state
        done = False #infinite horizon
        return next_state, (payoff_1, payoff_2), done

    
###/
# Example usage
if __name__ == "__main__":
    env = RepeatedPrisonersDilemmaEnv()
    state = env.reset()
    print(state)  # Initial state
    next_state, payoffs, done = env.step(C, C)
    print(next_state)  # Next state after actions
    print(payoffs)  # Payoffs for both players
    print(done)  # Whether the episode is done
    ####
