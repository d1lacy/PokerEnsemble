from pypokerengine.api.game import setup_config, start_poker
from custom_player import CustomPlayer
from Others.naiveplayer import NaivePlayer
from CFR.CFR import CFRPlayer
from alphapoker.alphaplayer import AlphaPlayer
from evolutionary.evolutionplayer import EvolutionPlayer
from tqdm import tqdm

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)

config.register_player(name="CustomPlayer", algorithm=CustomPlayer())
config.register_player(name="NaivePlayer", algorithm=NaivePlayer())
# config.register_player(name="CFRPlayer", algorithm=CFRPlayer())
# config.register_player(name="AlphaPlayer", algorithm=AlphaPlayer(model="alphapoker/models/pretrain_data_50round_500games_naivplayer100_8epoch.pth.tar"))
# config.register_player(name="EvolutionPlayer", algorithm=EvolutionPlayer())

num_games = 10
winner_count = {}
for i in tqdm(range(num_games)):
    game_result = start_poker(config, verbose=False)
    winner = None
    max_stack = 0
    for player in game_result['players']:
        if player['stack'] > max_stack:
            max_stack = player['stack']
            winner = player['name']
    if winner not in winner_count:
        winner_count[winner] = 0
    winner_count[winner] += 1
    print(f"Winner: {winner}, Stack: {max_stack}")

print("Winner count:", winner_count)
