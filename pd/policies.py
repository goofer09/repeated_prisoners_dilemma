from payoff import C,D
from dataclasses import dataclass
from typing import Protocol

State=tuple[int,int] |None

def validate_player_id(player_id)->None:
    if player_id not in (1,2):
        raise ValueError("player_id must be either 1 or 2")



def validate_action(action)->None:
    if action not in (C,D):
        raise ValueError("action must be either C or D")



def validate_state(state:State)->None:
    if state is None:
        return
    if len(state)!=2:
        raise ValueError("state must be None or tuple(a1,a2)")
    validate_action(state[0])
    validate_action(state[1])



class Policy(Protocol):

    name:str
    def act(self, state:State,player_id:int)->int:
         """Return an action for the given state and player."""

@dataclass(frozen=True)
class AlwaysC:
    name:str="AlwaysC"

    def act(self,state:State,player_id:int)->int:
        validate_player_id(player_id)
        validate_state(state)
        return C

@dataclass(frozen=True)
class AlwaysD:
    name:str="AlwaysD"

    def act(self,state:State,player_id:int)->int:
        validate_player_id(player_id)
        validate_state(state)
        return D


@dataclass(frozen=True)
class TitForTat:

    name:str="TitForTat"

    def act(self,state:State,player_id:int)->int:
        validate_player_id(player_id)
        validate_state(state)
        
        if state == None: 
            return C

        a1_prev, a2_prev=state
        opponent_last =a2_prev if player_id==1 else a1_prev
        validate_action(opponent_last)
        return opponent_last
    
    ###Name error mistake , I was initializing a1_prev and a2_prev , as state = a1,a2 , but Python wants to read them before they exist. Thus the correct direction was implemented