import torch
import socket
import struct
import time
from typing import *
from io import BytesIO
from functools import partial
from tqdm.auto import tqdm as orig_tqdm
from model import Brain, DQN, AuxNet
from collections import defaultdict

def filtered_trimmed_lines(lines):
    return filter(lambda l: l, map(lambda l: l.strip(), lines))

def create_model_from_config(state_dict, device="cpu", eval=False):
    config = state_dict["config"]
    version = config["control"]["version"]
    use_transformer = config["control"].get("use_transformer", False)
    aux_net_dims = config.get("aux", {}).get("dims", [4])
    if use_transformer:
        pass
    else:
        mortal = Brain(version=version, **config["resnet"]).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet(aux_net_dims).to(device) if not eval else None
    mortal.load_state_dict(state_dict["mortal"])
    dqn.load_state_dict(state_dict["current_dqn"])    
    aux_net.load_state_dict(state_dict["aux_net"])
    
    if eval:
        mortal.eval()
        dqn.eval()
    return mortal, dqn, aux_net

def player_state_to_human_tiles_info(player_state):
    # print(player_state)
    print(player_state.tehai)
    print(player_state.brief_info())
    # Convert tehai to string representation
    # tiles = []
    # for i, count in enumerate(player_state.tehai):
    #     if count > 0:
    #         suit = 'm' if i < 9 else 's' if i < 18 else 'p'
    #         tile_number = (i % 9) + 1
    #         tiles.extend([str(tile_number)] * count)
    #     if i == 8 or i == 17:
    #         tiles.append(suit)
    # if tiles[-1] in 'msp':
    #     tiles.append('#')
    
    # # Convert fuuro to string representation
    # melds = []
    # for meld in player_state.fuuro_overview[0]:  # Assuming player_id is 0
    #     meld_str = ''.join(str(t.as_usize() % 9 + 1) for t in meld)
    #     meld_str += 'msp'[meld[0].as_usize() // 9]
    #     melds.append(meld_str.upper())
    
    # # Convert dora indicators to dora tiles
    # dora_tiles = []
    # for indicator in player_state.dora_indicators:
    #     dora_tile = (indicator.as_usize() + 1) % 34
    #     suit = 'm' if dora_tile < 9 else 's' if dora_tile < 18 else 'p'
    #     tile_number = (dora_tile % 9) + 1
    #     dora_tiles.append(f"{tile_number}{suit}")
    
    # return {
    #     "HumanTiles": ''.join(tiles),
    #     "HumanMelds": melds,
    #     "HumanDoraTiles": ' '.join(dora_tiles),
    #     "HumanTargetTile": "",  # This information is not available in PlayerState
    #     "IsTsumo": player_state.last_self_tsumo is not None
    # }


