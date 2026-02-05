C=1
D=0

T=4
R=3
P=2
S=0


PD_PAYOFFS = {
    (C, C): (R, R),
    (C, D): (S, T),
    (D, C): (T, S),
    (D, D): (P, P),
}

def get_payoff(action1, action2):
    """Get the payoffs for two actions in a Prisoner's Dilemma game.

    Args:
        action1: The action of player 1 (C or D).
        action2: The action of player 2 (C or D).

    Returns:
        A tuple containing the payoffs for player 1 and player 2.
    """
    if (action1, action2) not in PD_PAYOFFS:
        raise ValueError("Invalid actions. Actions must be C (cooperate) or D (defect).")
    
    if action1 not in (C, D) or action2 not in (C, D):
        raise ValueError("Actions must be either C (cooperate) or D (defect).")
    
    if (action1, action2) in PD_PAYOFFS:
        return PD_PAYOFFS[(action1, action2)]
    


 # Example usage
if __name__ == "__main__":
    # Example usage here
    print(get_payoff(C, D))