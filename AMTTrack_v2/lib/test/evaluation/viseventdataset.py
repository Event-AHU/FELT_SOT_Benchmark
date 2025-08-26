import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

# TODO: class-VisEventDataset
class VisEventDataset(BaseDataset):
    def __init__(self, split):
        super().__init__()
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.visevent_path,  split)
        else:
            self.base_path = os.path.join(self.env_settings.visevent_path, 'train')
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        if split == 'val' or split == 'train':
            with open('{}/{}.txt'.format(self.env_settings.dataspec_path, split)) as f:
                seq_ids = f.read().splitlines()
            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64).reshape(-1, 4)

        aps_seq_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'vis_imgs')
        aps_frame_list = [frame for frame in os.listdir(aps_seq_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        aps_frame_list.sort(key=lambda f: int(f[-8:-4]))
        aps_frame_list = [os.path.join(aps_seq_path, frame) for frame in aps_frame_list]

        dvs_seq_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'event_imgs')
        dvs_frame_list = [frame for frame in os.listdir(dvs_seq_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        dvs_frame_list.sort(key=lambda f: int(f[-8:-4]))
        dvs_frame_list = [os.path.join(dvs_seq_path, frame) for frame in dvs_frame_list]

        return Sequence(sequence_name, aps_frame_list, 'VisEvent', ground_truth_rect, dvs_frame_list=dvs_frame_list) 