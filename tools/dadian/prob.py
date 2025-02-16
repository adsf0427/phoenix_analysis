import numpy as np
import torch
import json
import os
import sys  

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.dirname(current_dir))
from libriichi_analysis.state import PlayerState
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
net = nn.Sequential(b1, nn.Conv1d(256, 16, kernel_size=1, stride=1), nn.Flatten(), nn.Linear(544, 34), nn.ReLU(), nn.Dropout(0.3), nn.Linear(34, 1))
model = net
model.load_state_dict(torch.load('dadian.pth', map_location=device))
model = model.to(device)
model.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device('cpu')


# Print the number of parameters


state = None
def main():
    
    print(f"Number of trainable parameters: {count_parameters(model):,}")
    player_id = 0
    init_player_state(player_id)
    file_name = '../../input/2.json'

    f = open(file_name, 'r', encoding='UTF-8')
    file_content = f.readlines()
    rows = len(file_content)
    for line_n in range(0, rows):
        line = file_content[line_n]
        logits = update(line)

        if logits is not None:
            # pass
            print(line_n, line, logits)
            print('------------\n')



def init_player_state(player_id):
    global state
    state = PlayerState(player_id)

def update(line):
    global state
    global model
    global device
    
    result = state.update(line)
    current_line = json.loads(line)
    if current_line['type'] in ['reach_accepted', 'reach', 'start_kyoku', 'end_kyoku', 'hora', 'ryukyoku', 'start_game', 'end_game']:
        return None
    
    if current_line['type'] == 'tsumo' and current_line['actor'] != state.player_id:
        return None

    obs, masks = state.encode_obs(4, False)
    obs = torch.tensor(obs).unsqueeze(0).to(dtype=torch.float32, device=device)
    masks = torch.tensor(masks).unsqueeze(0).to(dtype=torch.float32, device=device)
    logits = model(obs)
    
    return logits.detach().cpu().numpy()[0]

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass