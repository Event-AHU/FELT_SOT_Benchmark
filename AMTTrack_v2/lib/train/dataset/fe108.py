import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader, opencv_loader
from lib.train.admin import env_settings
import scipy.io as scio

# TODO: class-fe108
class Fe108(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
    # def __init__(self, root=None, image_loader=opencv_loader, split=None, seq_ids=None, data_fraction=None):

        root = env_settings().got10k_dir if root is None else root
        super().__init__('Fe108', root, image_loader)

        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            if split == 'train':
                file_path = os.path.join(self.root,'train.txt')
            elif split == 'val':
                file_path = os.path.join(self.root, 'val.txt')
            ###################################################
            elif split == 'test':
                file_path = os.path.join(self.root, 'test.txt')
            ###################################################
            else:
                raise ValueError('Unknown split name')
            seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

    def get_name(self):
        return 'fe108'

    def _get_sequence_list(self):
        with open(os.path.join(self.root,'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_aps_sequence_path(self, seq_id):  # get aps_dir_path
        return os.path.join(self.root, self.sequence_list[seq_id], "aps")

    def _get_dvs_sequence_path(self, seq_id):  # get dvs_dir_path
        return os.path.join(self.root, self.sequence_list[seq_id], 'dvs')

    def _get_stack_sequence_path(self, seq_id):  # get stack_dir_path
        return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id]+'_stack')

    def _get_grountgruth_path(self, seq_id):  # get groundturth_dir_path
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        bbox_path = self._get_grountgruth_path(seq_id)
        # print(bbox_path)
        bbox = self._read_bb_anno(bbox_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        # return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
        return {'bbox': bbox, 'valid': valid, 'visible': visible, }

    def _get_frame_path(self, seq_path, frame_id):  # get rgb or event or stack frames path
        if os.path.exists(os.path.join(seq_path, 'frame{:04}.png'.format(frame_id))):
            return os.path.join(seq_path, 'frame{:04}.png'.format(frame_id))    # frames start from 0
        else:
            return os.path.join(seq_path, 'frame{:04}.bmp'.format(frame_id))    # some image is bmp

    def _get_frame(self, seq_path, frame_id):  # get a rgb or event or stack frame
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_event_voxel_sequence_path(self, seq_id):  # get events_voxel_dir_path
        return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id]+ "_voxel")

    def _get_event_voxel(self, seq_path, frame_id):  # get a events voxel
        frame_event_list = []
        for f_id in frame_id:
            event_frame_file = os.path.join(seq_path, 'frame{:04}.mat'.format(f_id))
            if os.path.getsize(event_frame_file) == 0:  # 如果文件大小为0，表示文件为空或不存在有效数据
                event_features = np.zeros(4096, 19)  # voxel.mat文件为空时，创建一个形状为 (4096, 19)的全零数组作为事件特征，表示没有事件发生or数据无效
                # need_data = [np.zeros([4096, 3]), np.zeros([4096, 16])]
            else:
                mat_data = scio.loadmat(event_frame_file)  # 加载mat文件数据
                # need_data = [mat_data['coor'], mat_data['features']]
                event_features = np.concatenate((mat_data['coor'], mat_data['features']), axis=1)  # 合并坐标和特征 concat coorelate and features (x,y,z, feauture32/16)
                
                if np.isnan(event_features).any():  # 处理NaN值
                    # 如果合并后的事件特征中包含NaN值，将其替换为全零数组，并打印警告信息
                    event_features = np.zeros(4096, 19)                     
                    print(event_frame_file, 'exist nan value in voxel.')

            frame_event_list.append(event_features)  # 将处理后的事件特征，添加到事件帧列表中
        return frame_event_list

    def get_frames(self, seq_id, frame_ids, anno=None):
        ###################################################################################
        # aps
        aps_seq_path = self._get_aps_sequence_path(seq_id)  # aps
        aps_frame_list = [self._get_frame(aps_seq_path, f_id) for f_id in frame_ids]  # aps中的对应ids的图像的对象列表
        ###################################################################################

        ###################################################################################
        # dvs
        dvs_seq_path = self._get_dvs_sequence_path(seq_id)  # dvs
        dvs_frame_list = [self._get_frame(dvs_seq_path, f_id) for f_id in frame_ids]  # dvs中的对应ids的事件图像的对象列表
        ###################################################################################
        
        ###################################################################################
        # stack
        # stack_seq_path = self._get_stack_sequence_path(seq_id)  # stack
        # stack_frame_list = [self._get_frame(stack_seq_path, f_id) for f_id in frame_ids]  # stack中的对应ids的事件图像的对象列表
        ###################################################################################
    
        ###################################################################################
        # voxel
        # event_voxel_seq_path = self._get_event_voxel_sequence_path(seq_id)  # voxel
        # event_voxel_list = self._get_event_voxel(event_voxel_seq_path, frame_ids)  # voxel中对应ids的事件voxel的对象列表
        ###################################################################################

        # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        # return aps_frame_list, anno_frames, object_meta  # rgb
        # return dvs_frame_list, anno_frames, object_meta  # event
        # return stack_frame_list, anno_frames, object_meta  # stack
        return aps_frame_list, dvs_frame_list, anno_frames, object_meta  # rgb + event
        # return aps_frame_list, event_voxel_list, anno_frames, object_meta   # rgb + event_voxel
        # return aps_frame_list, dvs_frame_list, event_voxel_list, anno_frames, object_meta  # rgb + event + event_voxel