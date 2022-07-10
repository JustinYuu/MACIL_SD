import torch.utils.data as data
import numpy as np
from utils import process_feat, process_test_feat
import os


class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False):
        self.modality = args.modality
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.normal_flag = '_label_A'
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        self.audio_list = list(open(self.audio_list_file))

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0
        # print(len(self.audio_list))
        # print(len(self.list))
        f_v = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        f_a = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
        if self.transform is not None:
            f_v = self.transform(f_v)
            f_a = self.transform(f_a)
        if self.test_mode:
            return f_v, f_a
        else:
            f_v = process_feat(f_v, self.max_seqlen, is_random=False)
            f_a = process_feat(f_a, self.max_seqlen, is_random=False)
            return f_v, f_a, label

    def __len__(self):
        return len(self.list)
