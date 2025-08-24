import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

# TODO: class-Fe108Dataset
class FE108Dataset(BaseDataset):

    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.fe108_path,  split)
        else:
            self.base_path = os.path.join(self.env_settings.fe108_path, 'train')

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
        ###################################################################################
        # groundtruth
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64).reshape(-1, 4)
        ###################################################################################
        
        ###################################################################################
        # aps
        aps_seq_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'aps')
        aps_frame_list = [frame for frame in os.listdir(aps_seq_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        aps_frame_list.sort(key=lambda f: int(f[-8:-4]))
        aps_frame_list = [os.path.join(aps_seq_path, frame) for frame in aps_frame_list]
        ###################################################################################

        ###################################################################################
        # dvs
        dvs_seq_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'dvs')
        dvs_frame_list = [frame for frame in os.listdir(dvs_seq_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        dvs_frame_list.sort(key=lambda f: int(f[-8:-4]))
        dvs_frame_list = [os.path.join(dvs_seq_path, frame) for frame in dvs_frame_list]
        ###################################################################################
        
        ###################################################################################
        # stack
        # stack_seq_path = '{}/{}/{}'.format(self.base_path, sequence_name, sequence_name+'_stack')
        # stack_frame_list = [frame for frame in os.listdir(stack_seq_path) if frame.endswith(".png") or frame.endswith(".bmp") ]
        # stack_frame_list.sort(key=lambda f: int(f[-8:-4]))
        # stack_frame_list = [os.path.join(stack_seq_path, frame) for frame in stack_frame_list]
        ###################################################################################

        ###################################################################################
        # voxel
        # event_voxel_seq_path = '{}/{}/{}'.format(self.base_path, sequence_name, sequence_name+'_voxel')
        # event_voxel_list = [frame for frame in os.listdir(event_voxel_seq_path) if frame.endswith(".mat")]
        # event_voxel_list.sort(key=lambda f: int(f[-8:-4]))
        # event_voxel_list = [os.path.join(event_voxel_seq_path, frame) for frame in event_voxel_list]
        ###################################################################################
        
        # return Sequence(sequence_name, aps_frame_list, 'FE108', ground_truth_rect)  # rgb
        # return Sequence(sequence_name, dvs_frame_list, 'FE108', ground_truth_rect)  # event
        # return Sequence(sequence_name, stack_frame_list, 'FE108', ground_truth_rect)  # stack
        return Sequence(sequence_name, aps_frame_list, 'FE108', ground_truth_rect, dvs_frame_list=dvs_frame_list)  # rgb + event
        # return Sequence(sequence_name, aps_frame_list, 'FE108', ground_truth_rect, event_voxel_list=event_voxel_list)  # rgb + event_voxel
        # return Sequence(sequence_name, aps_frame_list, 'FE108', ground_truth_rect, dvs_frame_list=dvs_frame_list, event_voxel_list=event_voxel_list)  # rgb + event + event_voxel