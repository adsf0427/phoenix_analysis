import argparse
import numpy as np
import torch
import torch.nn.functional as F
import json, gzip, requests
from libriichi.state import PlayerState
from libriichi.mjai import Bot
from common import *
import sys
import os
from collections import defaultdict
from ako import init_houjuu, process_input, cleanup_houjuu
import random
from collections import OrderedDict
import hashlib
from helper import analysis_tehai
import tools.weici.prob as weici
import tools.hu.prob as agari
import tools.dadian.prob as dadian
from model import Brain, DQN, AuxNet
from engine import MortalEngine

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# set_seed(42)  # 在主函数开始时调用

# rng = random.Random(hash(json.dumps(16324678434646, sort_keys=True)))
def actions_to_hai_str(actions):
    pass



def action_to_hai_index(action):
    if action < 34:
        return action
    else:
        aka_id = action - 34
        return aka_id * 9 + 4
    
def hai_index_to_str(hai_index):
    MJAI_PAI_STRINGS = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",  # m
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",  # p
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",  # s
        "E", "S", "W", "N", "P", "F", "C",  # z
        "5mr", "5pr", "5sr",  # aka
        "?"  # unknown
    ]
    
    if 0 <= hai_index < len(MJAI_PAI_STRINGS):
        return MJAI_PAI_STRINGS[hai_index]
    else:
        return "?"  # Return unknown for invalid indices
    
boltzmann_temp = 1.0
def gen_meta(input_dict, masks: torch.Tensor, ako_result, helper_result, rank_prob, agari_info, dadian_predict):
    q_out: torch.Tensor = input_dict['q_values'].cpu()
    masks = masks.cpu()
    q_out = q_out.squeeze(0)  # Remove the first dimension if it's 1
    
    # Ensure masks is also squeezed to match q_out's dimensions
    masks = masks.squeeze(0)
    
    assert masks.any()
    
    can_discard = masks[:36].any()
    logits = q_out / boltzmann_temp
    logits.masked_fill(~masks, -torch.inf)
    valid_logits = logits[masks]
    k = 15
    top_k_actions = torch.argsort(valid_logits, descending=True)[:k]
    
    # print(f"masks: {masks}")
    top_k_actions = torch.nonzero(masks).squeeze(1)[top_k_actions]
    
    # print(f"shape of actions: {top_3_actions.shape}, actions: {top_3_actions}")
    # print(f"can_discard: {can_discard}")
    
    meta_dict = defaultdict(list)
    
    if rank_prob is not None:
        meta_dict['rank_prob'] = rank_prob.tolist()
        
    # print(agari_info)
    if agari_info is not None:
        meta_dict['和率2'] = [f"{item[0]} {item[1]:.2%}" for item in agari_info.items()]        

    if dadian_predict is not None:
        meta_dict['打点预测'] = dadian_predict.tolist()

    ako_dict = None
    if ako_result:
        ako_dict = {}
        try:
            houjuu_prob_list = []    
            agari_prob_list = []        
            kr_result_list = []
            seed = int(hashlib.md5(ako_result.encode()).hexdigest(), 16)
            rng = random.Random(seed)
            ako_result = json.loads(ako_result, object_pairs_hook=OrderedDict)
            for item in ako_result:
                try:
                    pai = item['moves'][0]['pai']
                    houjuu_prob = item['review']['total_houjuu_hai_prob_now']                    
                    kr_result = item['review']['kyoku_result_prob']
                    scale = 0.95 + (rng.random() * 0.1)
                    houjuu_prob_list.append((pai, houjuu_prob * scale))
                    
                    kr_result[0] = kr_result[0] * scale
                    kr_result[1] = kr_result[1] * scale
                    kr_result[2] = kr_result[2] * 1
                    kr_result[3] = kr_result[3] * (2 - scale)
                    kr_result[4] = kr_result[4] * (2 - scale)
                    agari_prob = kr_result[0]
                    agari_prob_list.append((pai, agari_prob))
                    kr_result_list.append((pai, kr_result))
                    ako_dict[pai] = {
                        '铳率': houjuu_prob * scale,
                        '和率': agari_prob,
                        '单局预测': kr_result,
                    }                    
                except KeyError:
                    continue
            
            # Sort the list by total_houjuu_hai_prob_now in descending order
            houjuu_prob_list.sort(key=lambda x: x[1], reverse=True)
            meta_dict['铳率'] = [f"{item[0]} {item[1]:.2%}" for item in houjuu_prob_list]
            agari_prob_list.sort(key=lambda x: x[1], reverse=True)
            meta_dict['和率'] = [f"{item[0]} {item[1]:.2%}" for item in agari_prob_list]
            kr_result_list.sort(key=lambda x: x[1][0], reverse=True)
            meta_dict['单局预测'] = [f"{item[0]} {' '.join([f'{x:.2%}' for x in item[1]])}" for item in kr_result_list]
        except Exception as e:
            print(f"Error processing ako_result: {e}")
            print(ako_result)      
    
    helper_analysis_list = []
    avg_agari_probs = []
    avg_agari_weights = []
    # helper_analysis_list = [None] * 46
    for action in top_k_actions:
        hai_str_with_aka = hai_index_to_str(action)
        if hai_str_with_aka == '?':
            continue
        hai_index = action_to_hai_index(action)
        hai_str_deaka = hai_index_to_str(hai_index)
        discard_analysis = {}
        discard_analysis['action_index'] = action.item()
        discard_analysis['discard_tile'] = hai_str_with_aka

        if helper_result:
            assert can_discard
            meta_dict['shanten'] = helper_result['shanten']
            results14 = helper_result['results14']
            incShantenResults14 = helper_result['incShantenResults14']
            helper_analysis = None
            for result in results14:
                if result['discardTile'] == hai_index:
                    helper_analysis = result
                    break
            
            # Process incShantenResults14
            if not helper_analysis:
                for result in incShantenResults14:
                    if result['discardTile'] == hai_index:
                        helper_analysis = result
                        break  
                    
            discard_analysis["round_score"] = q_out[action].item() * 10000
                    
            if helper_analysis:
                waits = helper_analysis['result13']['waits']
                waits = waits.keys()
                waits = [int(wait) for wait in waits]
                waits = sorted(waits)
                waits = [hai_index_to_str(wait) for wait in waits]
                waits = " ".join(waits)
                discard_analysis.update({
                    "shanten": helper_analysis['result13']['shanten'],
                    "winning_tile_type": helper_analysis['result13']['yakuTypes'],
                    "tile_efficiency": helper_analysis['result13']['mixedWaitsScore'],
                    # "向听前进后进张": helper_analysis['result13']['avgNextShantenWaitsCount'],
                    "improvement_tiles": helper_analysis['result13']['avgImproveWaitsCount'],
                    "estimated_points": helper_analysis['result13']['mixedRoundPoint'],
                    "riichi_benefit": helper_analysis['result13']['riichiPoint'],
                    "silent_benefit": helper_analysis['result13']['damaPoint'],
                    "improvement_tiles_list": waits,
                })
                # print(f"waits: {waits}")
            if ako_dict:
                if agari_info:
                    avg_agari_probs.append(agari_info[hai_str_deaka])
                    avg_agari_weights.append(logits[action])
                discard_analysis['houjuu_rate'] = ako_dict[hai_str_with_aka]['铳率']
                discard_analysis['winning_rate'] = ako_dict[hai_str_with_aka]['和率']
                if agari_info:
                    discard_analysis['winning_rate2'] = agari_info[hai_str_deaka]
                else:
                    discard_analysis['winning_rate2'] = 0.0
                discard_analysis['single_round_prediction'] = ako_dict[hai_str_with_aka]['单局预测']
              
        # helper_analysis_list[action] = discard_analysis
        helper_analysis_list.append(discard_analysis)
        
    avg_agari_weights = torch.softmax(torch.tensor(avg_agari_weights), dim=0).tolist()
    # Calculate weighted average of agari probabilities
    if avg_agari_probs and avg_agari_weights:
        avg_agari_prob = sum(p * w for p, w in zip(avg_agari_probs, avg_agari_weights))
        meta_dict['综合和率'] = avg_agari_prob
    
    meta_dict['analysis'] = helper_analysis_list
        
        
    masks = masks.squeeze(0) 
    
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.squeeze(0)

    mask_bits = 0
    

    for i, m in enumerate(masks):
        if m:
            mask_bits |= 1 << i
            for key, value in input_dict.items():
                if value.shape[-1] == masks.shape[-1]:
                    meta_dict[key].append(value[i].item())
                    
    for key, value in input_dict.items():
        if value.shape[-1] != masks.shape[-1]:
            meta_dict[key] = value.tolist()
    meta_dict['mask_bits'] = mask_bits
    
    ordered_meta_dict = OrderedDict()
    for key in ['向听', 'analysis', 'rank_prob', '打点预测', '综合和率', '铳率', '和率', '和率2', '单局预测', 'mask_bits']:
        ordered_meta_dict[key] = meta_dict[key]
    

    return ordered_meta_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze mahjong game states.")
    parser.add_argument("-m", "--state_file", help="Path to the state file", default='models/score_only/v4_score_0930.pth')
    # parser.add_argument("-p", "--player_id", help="Player ID", type=int, default=0)
    # parser.add_argument("-d", "--detail", action="store_true", help="Include detailed output")
    # parser.add_argument("-i", "--input", help="Path to the input folder", default='input')
    # parser.add_argument("-o", "--output_folder", help="Path to the output folder", default='output')
    return parser.parse_args()

def get_reactoiins(mjai, player_id):
    post_data = {'data': mjai.split('\n')}
    data = json.dumps(post_data, separators=(',', ':'))
    compressed_data = gzip.compress(data.encode('utf-8'))

    # @app.route("/engine_4p/<string:engine_name>/<string:player_id>", methods=['GET'])
    result = requests.get('http://122.51.149.46:9875/engine_4p' + '/0/' + str(player_id), data=compressed_data)

    reactions = json.loads(gzip.decompress(result.content))['data']
    yield from reactions
    # return reactions

def main():
    try:
        player_id = int(sys.argv[-1])
        assert player_id in range(4)
    except:
        sys.exit(1)    
    # args = parse_arguments()
    # player_id = args.player_id
    player_state = PlayerState(player_id)
    weici.init_player_state(player_id)
    dadian.init_player_state(player_id)
    state_file = 'models/score_only/v4_score_0930.pth'
    state_dict = torch.load(
        state_file, map_location=torch.device("cpu")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cfg = state_dict["config"]
    mortal, dqn, auxnet = create_model_from_config(state_dict, device=device, eval=False)
    mortal.eval()
    dqn.eval()   
    auxnet.eval()
    engine = MortalEngine(
            mortal,
            dqn,
            is_oracle=False,
            version=4,
            stochastic_latent=False,
            device=device,
            enable_amp=False,
            enable_quick_eval=False,
            enable_rule_based_agari_guard=False,
            name='mortal',
        )    
    bot = Bot(engine, player_id)    
    mortal_review = True
    
    online = False
    
    if online:
        with open('input/mjai.json', 'r') as f:
            mjai = f.read()
        reactions = get_reactoiins(mjai, player_id)
    else :
        reactions = iter([])

  
    ako_state = init_houjuu(player_id)
    # lines = list()
    # agari_rate = agari.cal_agari_rate(player_id, lines)
    for line_id, line in enumerate(filtered_trimmed_lines(sys.stdin)):
        result = player_state.update(line)
        if online:
            reaction = next(reactions)
        else:
            reaction = bot.react(line)
        # print(reaction, flush=True)
        # continue
        
        rank_prob = weici.update(line)
        dadian_predict = dadian.update(line)
        event = json.loads(line)
        # Temporarily redirect stdout to suppress output from process_input
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            ako_result = process_input(ako_state, line, agari_and_houjuu=True)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

        obs, masks = player_state.encode_obs(4, False)
        obs = torch.tensor(obs).unsqueeze(0).to(dtype=torch.float32, device=device)
        masks = torch.tensor(masks).unsqueeze(0).to(dtype=torch.bool, device=device)
        can_act = masks.any()
        can_discard = masks[0][:36].any()
        # if reaction:
        #     print(reaction, flush=True)
        # else:
        #     print('{"type":"none","meta":{"mask_bits":0}}', flush=True)        
        if not can_act:
            # output_str += json.dumps(event, ensure_ascii=False) + '\n'
            # print(json.dumps(event, ensure_ascii=False), flush=True)
            print('{"type":"none","meta":{"mask_bits":0}}', flush=True)
            continue


        helper_result = None
        if can_discard:
            try:
                tile_str = player_state.tile_string()
                # print(tile_str)
                # print(f"tiles_seen: {player_state.tiles_seen}")
                # print(f"dora_tiles: {player_state.dora_tiles}")
                # print(f"bakaze: {player_state.bakaze}")
                # print(f"jikaze: {player_state.jikaze}")
                # print(f"is_parent: {player_state.bakaze == player_state.jikaze}")
                # print(f"discarded_tiles: {player_state.discarded_tiles}")
                other_info = {
                    'DoraTiles': player_state.dora_tiles,
                    'RoundWindTile': player_state.bakaze,
                    'SelfWindTile': player_state.jikaze,
                    'IsParent': player_state.bakaze == player_state.jikaze,
                    'DiscardTiles': player_state.discarded_tiles,
                    'LeftTiles34': player_state.left_tiles,
                    'LeftDrawTilesCount': player_state.tiles_left,
                }
                helper_result = analysis_tehai(tile_str, other_info)
                
                # print(helper_result)
            except BaseException as e:
                print(f"Caught panic: {e}")
                continue
    

        # with torch.no_grad():
        #     phi = mortal(obs)
        #     q_out = dqn(phi, masks)
        
        # if detail:
        #     (
        #         next_rank_logits,
        #         kyoku_points,
        #         houjuu_out,
        #         agari_out,
        #         houjuu_points,
        #         agari_points,
        #     ) = auxnet(phi)  
        # else:
        #     next_rank_logits, = auxnet(phi)  
        
    
        # next_rank_probs = F.softmax(next_rank_logits, dim=-1) 

        # data_dict = {}  

        # if detail:
        #     houjuu_prob = torch.sigmoid(houjuu_out)
        #     agari_prob = torch.sigmoid(agari_out)
        #     data_dict = {
        #         'next_rank_probs': next_rank_probs.detach().cpu(),
        #         'kyoku_points': kyoku_points.detach().cpu(),
        #         'houjuu_prob': houjuu_prob.detach().cpu(),
        #         'agari_prob': agari_prob.detach().cpu(),
        #         'houjuu_points': houjuu_points.detach().cpu(),
        #         'agari_points': agari_points.detach().cpu(),
        #         'q_values': q_out.detach().cpu(),
        #     }
        
        
        q_values = json.loads(reaction)["meta"]["q_values"]
        q_values = torch.tensor(q_values)
        full_q_values = torch.full((masks.shape[-1],), -float('inf'))
        full_q_values[masks[0]] = q_values
        q_values = full_q_values
        data_dict = {
            'q_values': q_values,
        }

        if can_act:
            # agari_info = agari_rate.get(line_id, None)
            agari_info = None
            event['meta'] = gen_meta(data_dict, masks, ako_result, helper_result, rank_prob, agari_info, dadian_predict)  
            # output_str += json.dumps(event, ensure_ascii=False) + '\n'
            reaction = json.loads(reaction)   
            # print(json.dumps(event["meta"], ensure_ascii=False, separators=(',', ':')), flush=True)         
            reaction["analysis"] = event["meta"]["analysis"]
            reaction["rank_prob"] = event["meta"]["rank_prob"]
            print(json.dumps(reaction, ensure_ascii=False, separators=(',', ':')), flush=True)
            # print(json.dumps(event, ensure_ascii=False), flush=True)
            

    cleanup_houjuu(ako_state)
    extra_data = {
        "model_tag": 'coach',
        "phi_matrix": [[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],],
    } 
    print(json.dumps(extra_data), flush=True)    
            



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass