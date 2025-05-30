from CFR.CFR import CFRPlayer
from evolutionary.evolutionplayer import EvolutionPlayer
from Others.raise_player import RaisedPlayer
from Others.naiveplayer import NaivePlayer
from pypokerengine.players import BasePokerPlayer
from alphapoker.alphaplayer import AlphaPlayer
import numpy as np



class CustomPlayer(BasePokerPlayer):
    
    def __init__(self):
        # initialize ensemble of players
        self.players = [EvolutionPlayer(), CFRPlayer(), AlphaPlayer(model="alphapoker/models/both_ofem.pth.tar"), NaivePlayer()]
        self.names = ["Evolutionary", "CFRPlayer", "AlphaPlayer", "Naive"]
        self.n = len(self.players)
        
        # Initialize counts and means for each player
        self.counts = np.zeros(self.n)
        self.sums = np.zeros(self.n)
        self.sq_sums = np.zeros(self.n)

        # initialize player variables (which one plays each round)
        self.curr_player = 0
        self.curr_index = 0

        # initialize round count and stack to track new rounds
        self.round_count = 0
        self.stack = 1000
        self.played = False
        self.new_game = True


    def declare_action(self, valid_actions, hole_card, round_state):

        # set up for game start:
        uuid = self.uuid
        for player in self.players:
            player.uuid = uuid
    
        state_rc = round_state['round_count']
        if state_rc <= 1 and self.new_game:
            self.counts = np.zeros(self.n)
            self.sums = np.zeros(self.n)
            self.sq_sums = np.zeros(self.n)
            self.round_count = 0
            self.stack = 1000
            self.played = False
            self.new_game = False
            print("New game started")

        if state_rc == 2:
            self.new_game = True
        

        # handle a new round
        if self.round_count != state_rc:
            # update previous round (if any)
            if self.played:
                for seat in round_state['seats']:
                    if seat['uuid'] == self.uuid:
                        end_stack = seat['stack']

                reward = end_stack - self.stack
                self.stack = end_stack

                # update
                print(f"Player {self.names[self.curr_index]}: {reward}")
                i = self.curr_index
                self.counts[i] += 1
                self.sums[i] += reward
                self.sq_sums[i] += reward**2

            # get a new player to play the new round
            if state_rc < self.n + 1:
                self.curr_index = state_rc - 1

            else:
                # Thompson Sampling
                samples = []
                for i in range(self.n):
                    if self.counts[i] == 0:
                        samples.append(np.inf)  # force exploration
                    else:
                        mean = self.sums[i] / self.counts[i]
                        var = (self.sq_sums[i] / self.counts[i]) - mean**2
                        std = np.sqrt(var + 1e-6)  # add small epsilon to avoid 0 std
                        samples.append(np.random.normal(mean, std))
                self.curr_index = int(np.argmax(samples))
            
            self.played = True
            self.round_count = state_rc
            
            self.curr_player = self.players[self.curr_index]

            print(f"\t **** Round {state_rc}: Player {self.names[self.curr_index]}")


        # return the action for the player being used in this round
        return self.players[self.curr_index].declare_action(valid_actions, hole_card, round_state)

    def receive_game_start_message(self, game_info):
        pass
        

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return CustomPlayer()