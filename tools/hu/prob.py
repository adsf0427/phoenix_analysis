import numpy as np
import torch
import json
from libriichi_analysis.state import PlayerState

def softmax(x):
    # exps = np.exp(x - np.max(x))
    # return exps / np.sum(exps)
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

dictjianxie = {0: '1m', 1: '2m', 2: '3m', 3: '4m', 4: '5m', 5: '6m', 6: '7m', 7: '8m', 8: '9m',
                           9: '1p', 10: '2p', 11: '3p', 12: '4p', 13: '5p', 14: '6p', 15: '7p', 16: '8p', 17: '9p',
                           18: '1s', 19: '2s', 20: '3s', 21: '4s', 22: '5s', 23: '6s', 24: '7s', 25: '8s', 26: '9s',
                           27: 'E', 28: 'S', 29: 'W', 30: 'N', 31: 'P', 32: 'F', 33: 'C', 34: '5mr', 35: '5pr', 36: '5sr'}


device = torch.device('cpu')
import torch.nn as nn
from torch.nn import functional as F

class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.bn3 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.bn3(self.conv3(X))

        Y += X

        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(*resnet_block(870, 256, 16))
net = nn.Sequential(b1, nn.Conv1d(256, 1, kernel_size=1, stride=1), nn.Flatten(), nn.Linear(34, 2))

model = net
model.load_state_dict(torch.load('hu.pth', map_location=device))
model = model.to(device)
model.eval()

def cal_agari_rate(player_id, lines):
    state = PlayerState(player_id)
    new_kyoku = 0
    agari_rates = {}

    file_content = lines
    
    for line_n, line in enumerate(file_content):
        result = state.update(line)
        current_line = json.loads(line)
        
        if current_line['type'] == 'start_kyoku':
            new_kyoku = line_n
            continue

        if not state.last_cans.can_discard:
            continue

        if result.can_agari:
            agari_rates[line_n] = {dictjianxie[i]: 1 for i in list(np.nonzero(np.array(state.tehai))[0])}
            continue
        
        if state.tiles_left == 0:
            agari_rates[line_n] = {dictjianxie[i]: 0 for i in list(np.nonzero(np.array(state.tehai))[0])}
            continue

        hu_dict = {}
        shoupai = list(np.nonzero(np.array(state.tehai))[0])
        
        for i in shoupai:
            state_new = PlayerState(player_id)
            for j in range(new_kyoku, line_n + 1):
                state_new.update(file_content[j])
            state_new.update(f'{{"type":"dahai","actor":{player_id},"pai":"{dictjianxie[i]}","tsumogiri":false}}')

            obs = state_new.encode_obs(4, False)
            obs = torch.tensor(obs[0]).unsqueeze(0).to(dtype=torch.float32, device=device)
            logits = model(obs)
            hu_prob = float(np.array(logits.softmax(-1).detach().cpu())[0, 1])
            hu_dict[dictjianxie[i]] = hu_prob

        agari_rates[line_n] = hu_dict

    return agari_rates

def main():
    player_id = 0
    file_path = '../../input/2.json'
    agari_rates = cal_agari_rate(player_id, file_path)
    
    with open(file_path, 'r') as f:
        file_content = f.readlines()

    for line_n, hu_list in agari_rates.items():
        print(line_n, file_content[line_n].strip(), hu_list)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass