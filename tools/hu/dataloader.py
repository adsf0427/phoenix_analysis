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

        for start_idx in range(0, len(self.file_list), 10):
            for j in range(0, 10):
                self.populate_buffer(self.file_list[start_idx + j:start_idx + j + 1][0])

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
        all_result = []

        current_state = [-2]
        for line_n in range(rows-1, -1, -1):
            line = file_content[line_n].decode().strip()
            current_line = json.loads(line)
            if current_line['type'] == 'start_kyoku' or current_line['type'] == 'end_kyoku' or current_line['type'] == 'start_game' or current_line['type'] == 'end_game':
                current_state = [-2]
            elif current_line['type'] == 'hora':
                sum_list = []
                for i in range(-2, 3):
                    test_line = json.loads(file_content[line_n + i].decode().strip())
                    if test_line['type'] == 'hora':
                        sum_list.append(test_line['actor'])

                        # if i != 0:
                        #     print(i)

                current_state = sum_list

            elif current_line['type'] == 'ryukyoku':
                current_state = [-1]

            all_result.append(current_state)


        all_result = all_result[::-1]


        for i in range(0, 4):
            for line_n in range(0, rows):
                line = file_content[line_n].decode().strip()
                reaction = self.bot[i].update(line)
                current_line = json.loads(line)
                if current_line['type'] != 'dahai':
                    continue

                if current_line['actor'] != i:
                    continue

                if reaction.can_agari:
                    continue

                if self.bot[i].tiles_left == 0:
                    continue

                assert all_result[line_n][0] != -2

                if all_result[line_n][0] == -1:
                    label = 0
                else:
                    if i in all_result[line_n]:
                        label = 1
                    else:
                        label = 0

                entry = [
                    self.bot[i].encode_obs(4, False)[0],
                    label
                ]
                self.buffer.append(entry)


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
