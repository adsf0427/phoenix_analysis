import numpy as np
import torch
import logging
logging.getLogger().setLevel(logging.INFO)
from glob import glob
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from libriichi.state import PlayerState

from dataloader import FileDatasetsIter, worker_init_fn

def main():
    logging.info('building file index...')
    file_list = []
    for pat in ['paipu\*.json.gz']:
        file_list.extend(glob(pat, recursive=True))
    file_list.sort(reverse=True)

    logging.info(f'file list size: {len(file_list):,}')
    # random.seed(2)
    random.shuffle(file_list)
    device = torch.device('cuda')

    bot1 = PlayerState(0)
    bot2 = PlayerState(1)
    bot3 = PlayerState(2)
    bot4 = PlayerState(3)
    bot = [bot1, bot2, bot3, bot4]

    file_data = FileDatasetsIter(
        file_list=file_list,
        bot=bot,
        num_epochs=1,
    )
    data_loader = iter(DataLoader(
        dataset=file_data,
        batch_size=8192,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    ))

    import torch.nn as nn
    from torch.nn import functional as F
    from torchsummary import summary

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
    model = model.to(device)
    learning_rate = 1e-4
    weight_decay = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1, eta_min=1e-6, last_epoch=-1, verbose=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    pb = tqdm(total=10000, desc='TRAIN', initial=0)
    batch = 0
    model.train()
    for obs, label in data_loader:
        train_loss = []
        train_accs = []
        mini_loss = []
        mini_accs = []

        obs = obs.to(dtype=torch.float32, device=device)
        labels = label.to(dtype=torch.long, device=device)
        logits = model(obs)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
        mini_loss.append(loss.item())
        mini_accs.append(acc)

        batch += 1
        if batch % 10 == 0:
            print(f"loss = {sum(mini_loss) / len(mini_loss):.5f}, acc = {sum(mini_accs) / len(mini_accs):.5f}, lr = {scheduler.get_last_lr()[0]:.7f}")
        if batch % 10 == 0:
            torch.save(model.state_dict(), 'hu.pth')

        pb.update(1)
    pb.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass