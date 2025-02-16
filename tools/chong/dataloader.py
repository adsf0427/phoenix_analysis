import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import json
import gzip


class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        file_list,
        bot,
        num_epochs = 1,
    ):
        super().__init__()
        self.file_list = file_list
        self.num_epochs = num_epochs
        self.iterator = None
        self.bot = bot
        # device = torch.device('cuda')


    def build_iter(self):
        # do not put it in __init__, it won't work on Windows

        for _ in range(self.num_epochs):
            yield from self.load_files()

    def load_files(self):
        # shuffle the file list for each epoch
        # random.shuffle(self.file_list)

        self.buffer = []

        for start_idx in range(0, len(self.file_list)):
            self.populate_buffer(self.file_list[start_idx:start_idx + 1][0])

            random.shuffle(self.buffer)
            yield from self.buffer[0:]
            del self.buffer[0:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        f = gzip.open(file_list, 'rb')
        # f = open(file_list, 'r', encoding='UTF-8')
        file_content = f.readlines()
        rows = len(file_content)
        # current_ju = 0
        all_result = [[], [], [], []]

        for i in range(0, 4):
            for line_n in range(0, rows):
                line = file_content[line_n].decode().strip()
                reaction = self.bot[i].update(line)

                all_result[i].append(np.array(self.bot[i].waits))

        all_result = np.array(all_result)

        for i in range(0, 4):
            if i == 0:
                label = all_result[1:, :]
            elif i == 1:
                label = all_result[[2, 3, 0], :]
            elif i == 2:
                label = all_result[[3, 0, 1], :]
            elif i == 3:
                label = all_result[:3, :]

            # label = (label.transpose(1, 0) * (2 ** np.arange(3))).sum(axis=1)

            # label_index = 0
            for line_n in range(0, rows):
                line = file_content[line_n].decode().strip()
                reaction = self.bot[i].update(line)

                if self.bot[i].last_cans.can_discard:
                    pass
                else:
                    continue

                a = self.bot[i].encode_obs(4, False)[0][132:324][range(3, 188, 8)].sum(axis=0) > 0
                b = self.bot[i].encode_obs(4, False)[0][327:519][range(3, 188, 8)].sum(axis=0) > 0
                c = self.bot[i].encode_obs(4, False)[0][522:714][range(3, 188, 8)].sum(axis=0) > 0

                if self.bot[i].last_tedashis2[2] != 40:
                    a[self.bot[i].last_tedashis2[2]] = True
                if self.bot[i].last_tedashis2[3] != 40:
                    a[self.bot[i].last_tedashis2[3]] = True
                    b[self.bot[i].last_tedashis2[3]] = True

                if (np.array(self.bot[i].last_tedashis3[0]) != 40).any():
                    a[np.nonzero(np.array(self.bot[i].last_tedashis3[0]) - 40)[0]]  = True
                if (np.array(self.bot[i].last_tedashis3[1]) != 40).any():
                    b[np.nonzero(np.array(self.bot[i].last_tedashis3[1]) - 40)[0]]  = True
                if (np.array(self.bot[i].last_tedashis3[2]) != 40).any():
                    c[np.nonzero(np.array(self.bot[i].last_tedashis3[2]) - 40)[0]]  = True

                a = np.logical_or(a, np.array(self.bot[i].last_tedashis4))
                b = np.logical_or(b, np.array(self.bot[i].last_tedashis4))
                c = np.logical_or(c, np.array(self.bot[i].last_tedashis4))

                entry = [
                    self.bot[i].encode_obs(4, False),
                    label[:, line_n, :].reshape(102).astype(int),
                    np.concatenate((a, b, c), axis=0)
                ]
                self.buffer.append(entry)
                # label_index += 1


    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
