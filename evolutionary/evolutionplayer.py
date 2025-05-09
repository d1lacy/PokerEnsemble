from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.utils.card_utils import gen_cards

class EvolutionPlayer(BasePokerPlayer):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = [0.8333073633481094, -0.45758701433365667, 0.7305590186946016, 0.3482318230293011, -0.0626087492349896, 0.3682071397713673, -0.32904816344230947, -0.2117890819574637]

    def evaluate_state(self, state_features):
        raw_score = sum(w * f for w, f in zip(self.weights, state_features))
        normalized_score = (raw_score + 8) / 16  # score ∈ [-8, +8] → [0, 1]
        return max(0.0, min(1.0, normalized_score))

    def declare_action(self, valid_actions, hole_card, round_state):
        features = extract_features(hole_card, round_state)
        score = self.evaluate_state(features)
        # print("valid_actions: ", valid_actions)
        # print(f"action score: {score}")
        if score > 0.7:
            action =  valid_actions[2]['action'] if any(a['action'] == 'raise' for a in valid_actions) else valid_actions[1]['action'] # raise
            # if any(a['action'] == 'raise' for a in valid_actions):
            #   amount = valid_actions[2]['amount'] 
            #   min_r = amount['min']
            #   max_r = amount['max']
            #   scaled_score = (score - 0.7) / 0.3
            #   scaled_score = max(0.0, min(scaled_score, 1.0))  # Clamp just in case
            #   amount = int(min_r + scaled_score * (max_r - min_r))
            #   amount = max(min_r, min(amount, max_r))
            # else: 
            #   amount = valid_actions[1]['amount']
        elif score > 0.3:
            action = valid_actions[1]['action'] # call
            # amount = valid_actions[1]['amount']
        else:
            action = valid_actions[0]['action'] # fold
            # amount = valid_actions[0]['amount']

        
        return action #, amount 

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

def extract_features(hole_card, round_state):
    strength = estimate_hand_strength(hole_card, round_state.get('community_card', []))  # ∈ [0,1]
    pot_amount = round_state['pot']['main']['amount']
    my_stack = round_state['seats'][round_state['next_player']]['stack']
    to_call = pot_amount
    pot_odds = pot_amount / (pot_amount + to_call + 1e-6)  # ∈ [0,1]
    stack_ratio = my_stack / 1000  # assuming initial stack is 1000
    board_texture = calculate_board_texture(round_state.get('community_card', []))  # ∈ [0,1]
    street_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    street_number = street_mapping.get(round_state['street'], 0) / 3  # normalize to [0,1]
    actions = round_state['action_histories'].get(round_state['street'], [])
    num_raises = sum(1 for action in actions if action['action'] == 'raise')
    norm_num_raises = min(num_raises, 3) / 3  # ∈ [0,1]
    opponent_aggression = calculate_opponent_aggression(round_state)  # ∈ [0,1]
    payout = strength * pot_amount
    norm_payout = min(payout, 1000) / 1000  # normalize to [0,1]

    return [strength, pot_odds, stack_ratio, board_texture,
            street_number, norm_num_raises, opponent_aggression, norm_payout]

def estimate_hand_strength(hole_card, community_card):
    nb_simulation = 100
    nb_player = 2
    win_rate = estimate_hole_card_win_rate(nb_simulation, nb_player, gen_cards(hole_card), gen_cards(community_card))
    return win_rate

def calculate_board_texture(community_card):
    suits = [card[-1] for card in community_card]
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values()) if suit_counts else 0
    return max_suit_count / 5

def calculate_opponent_aggression(round_state):
    aggression = 0
    for street, actions in round_state.get('action_histories', {}).items():
        for action in actions:
            if action['action'] == 'raise':
                aggression += 1
    total_actions = sum(len(actions) for actions in round_state.get('action_histories', {}).values())
    return aggression / (total_actions + 1e-6)