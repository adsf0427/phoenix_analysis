import numpy as np
import torch
import json
from libriichi.state import PlayerState
class MortalEngine:
    def __init__(
            self,
            brain=None,
            dqn=None,
            is_oracle=False,
            version=4,
            device=None,
            stochastic_latent=False,
            enable_amp=True,
            enable_quick_eval=False,
            enable_rule_based_agari_guard=False,
            name='NoName',
            boltzmann_epsilon=0,
            boltzmann_temp=1,
            top_p=1,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        # self.brain = brain.to(self.device).eval()
        # self.dqn = dqn.to(self.device).eval()
        self.is_oracle = False
        self.version = 4
        self.stochastic_latent = False

        self.enable_amp = True
        self.enable_quick_eval = False
        self.enable_rule_based_agari_guard = False
        self.name = name

        self.boltzmann_epsilon = 0
        self.boltzmann_temp = 1
        self.top_p = 1

    def react_batch(self, obs, masks, invisible_obs):
        # with (
        #     torch.autocast(self.device.type, enabled=self.enable_amp),
        #     torch.no_grad(),
        # ):
        return self._react_batch(obs, masks, invisible_obs)

    def _react_batch(self, obs, masks, invisible_obs):
        # obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        # invisible_obs = None
        b, c = masks.shape
        q_out = torch.zeros((b, 46), device=self.device)
        q_out = torch.masked_fill(q_out, ~masks, -torch.inf)
        actions = q_out.argmax(-1)
        is_greedy = torch.ones(b, dtype=torch.bool, device=self.device)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

def softmax(x):
    # exps = np.exp(x - np.max(x))
    # return exps / np.sum(exps)
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def main():
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
    net = nn.Sequential(b1, nn.Conv1d(256, 3, kernel_size=1, stride=1), nn.Flatten())

    model = net
    model.load_state_dict(torch.load('chong.pth'))
    model = model.to(device)
    model.eval()

    player_id = 0
    file_name = 'lizhi3.json'
    state = PlayerState(player_id)

    f = open(file_name, 'r', encoding='UTF-8')
    file_content = f.readlines()
    rows = len(file_content)
    for line_n in range(0, rows):
        line = file_content[line_n]
        result = state.update(line)
        current_line = json.loads(line)
        if current_line['type'] == 'start_kyoku' or current_line['type'] == 'end_kyoku' or current_line['type'] == 'hora' or current_line['type'] == 'ryukyoku' or current_line['type'] == 'start_game' or current_line['type'] == 'end_game':
            continue

        if state.last_cans.can_discard:
            pass
        else:
            continue

        if line_n == 106:
            print(line_n)

        a = state.encode_obs(4, False)[0][132:324][range(3, 188, 8)].sum(axis=0) > 0
        b = state.encode_obs(4, False)[0][327:519][range(3, 188, 8)].sum(axis=0) > 0
        c = state.encode_obs(4, False)[0][522:714][range(3, 188, 8)].sum(axis=0) > 0

        if state.last_tedashis2[2] != 40:
            a[state.last_tedashis2[2]] = True
        if state.last_tedashis2[3] != 40:
            a[state.last_tedashis2[3]] = True
            b[state.last_tedashis2[3]] = True

        if (np.array(state.last_tedashis3[0]) != 40).any():
            a[np.nonzero(np.array(state.last_tedashis3[0]) - 40)[0]] = True
        if (np.array(state.last_tedashis3[1]) != 40).any():
            b[np.nonzero(np.array(state.last_tedashis3[1]) - 40)[0]] = True
        if (np.array(state.last_tedashis3[2]) != 40).any():
            c[np.nonzero(np.array(state.last_tedashis3[2]) - 40)[0]] = True

        a = np.logical_or(a, np.array(state.last_tedashis4))
        b = np.logical_or(b, np.array(state.last_tedashis4))
        c = np.logical_or(c, np.array(state.last_tedashis4))

        obs = state.encode_obs(4, False)
        obs = torch.tensor(obs[0]).unsqueeze(0).to(dtype=torch.float32, device=device)
        logits = model(obs)
        logits = logits.masked_fill(torch.tensor(np.concatenate((a, b, c), axis=0)), -torch.inf)
        gailv = torch.sigmoid(logits)

        a1 = gailv[0, :34]
        b1 = gailv[0, 34:68]
        c1 = gailv[0, 68:]

        output = np.array(torch.max(torch.max(a1, b1), c1)[np.nonzero(np.array(state.tehai))].detach())

        print(line_n, line, np.array(output))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass