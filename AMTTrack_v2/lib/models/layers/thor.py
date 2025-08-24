import random
import torch
import torch.nn.functional as F
import math


class THOR_Wrapper():
    def __init__(self, net, st_capacity, lt_capacity, sample_interval, update_interval, lower_bound=0.90, score_threshold=0.7):
        self.net = net
        self.st_capacity = st_capacity
        self.lt_capacity = lt_capacity
        self.sample_interval = sample_interval
        self.update_interval = update_interval
        self.lower_bound = lower_bound
        self.score_threshold = score_threshold

        self.current_type = 'st'
        self.frame_no = 0
        self.zs = 64
        self.st_count = 0
        self.lt_count = 0
        self.st_update_count = 0
        self.lt_update_count = 0
        self.sample_count = 0

    def get_count(self):
        return self.st_count, self.lt_count

    def get_update_count(self):
        return self.st_update_count, self.lt_update_count, self.frame_no
    
    def get_sample_count(self):
        return self.sample_count, self.frame_no

    def setup(self, zi, ze):
        """initialize the short-term and long-term module"""
        self.st_module = ST_Module(net=self.net, z_capacity=self.st_capacity,)
        self.st_module.fill(zi, ze)
        self.lt_module = LT_Module(net=self.net, z_capacity=self.lt_capacity, lower_bound=self.lower_bound,)
        self.lt_module.fill(zi, ze)

    def update(self, zi, ze, pred_score):
        """update the short-term and long-term module"""
        self.frame_no += 1
        curr_zi = self.net.backbone._z_feat(zi.unsqueeze(0))
        curr_ze = self.net.backbone._z_feat(ze.unsqueeze(0))

        ###################################################################################
        # LT - clear
        if self.frame_no % 500 == 0:
            self.lt_module.clear()
        ###################################################################################

        ###################################################################################
        # NOTE: ST - update
        if pred_score >= self.score_threshold and self.frame_no % self.update_interval == 0:
            self.st_update_count += 1
            div_scale = self.st_module.update(zi, ze, curr_zi, curr_ze)
            # print('frame no: {}, div_scale: {}'.format(self.frame_no, div_scale))

            # NOTE: LT - update
            if self.frame_no % (self.update_interval * 2) == 0:
                update_flag = self.lt_module.update(zi, ze, curr_zi, curr_ze, div_scale=div_scale, time=self.frame_no)
                if update_flag:
                    self.lt_update_count += 1 
        ###################################################################################

        ###################################################################################
        if self.frame_no % self.sample_interval == 0 or self.frame_no == 1:
            self.sample_count += 1
            st_dynamic_zi, st_dynamic_ze = self.st_module.get_dynamic_z(1, resample_flag=True)  
            lt_dynamic_zi, lt_dynamic_ze = self.lt_module.get_dynamic_z(1, resample_flag=True) 
        else:
            st_dynamic_zi, st_dynamic_ze = self.st_module.get_dynamic_z(1, resample_flag=False)
            lt_dynamic_zi, lt_dynamic_ze = self.lt_module.get_dynamic_z(1, resample_flag=False)
        return torch.cat([lt_dynamic_zi, st_dynamic_zi], dim=1), torch.cat([lt_dynamic_ze, st_dynamic_ze], dim=1)
        ###################################################################################
        

class TemplateModule():
    def __init__(self, net, z_capacity, zs=64):
        self.net = net
        self.z_capacity = z_capacity
        self.zs = zs
        self.zi_raw_list = []
        self.ze_raw_list = []
        self.zi_list = []
        self.ze_list = []
        self.gram_matrix_zi = None
        self.gram_matrix_ze = None
        self.base_sim_zi = 0
        self.base_sim_ze = 0
        self.timestamp_list = []
        self.selected_i = []    
        self.selected_e = []    
        self.static_i = None
        self.static_e = None
        
    def __len__(self):
        return self.z_capacity

    def fill(self, zi, ze):
        self.static_i = zi
        self.static_e = ze
        """fill all slots in the memory with the given template"""
        # zi / ze.shape (1, 3, 128, 128)
        for _ in range(self.z_capacity): 
            self.zi_raw_list.append(zi)  
            self.ze_raw_list.append(ze)  
        self.gen_z_feature()  
        self.calculate_gram_matrix()
        self.base_sim_zi = self.gram_matrix_zi[0, 0]
        self.base_sim_ze = self.gram_matrix_ze[0, 0]
        self.timestamp_list = [0] * self.z_capacity
    
    def gen_z_feature(self):
        M = len(self.zi_raw_list)
        zi_list, ze_list = [], []
        for zi_raw, ze_raw in zip(self.zi_raw_list, self.ze_raw_list):
            zi = self.net.backbone._z_feat(zi_raw.unsqueeze(0))
            ze = self.net.backbone._z_feat(ze_raw.unsqueeze(0))
            zi_list.append(zi)
            ze_list.append(ze)
        self.zi_list = zi_list
        self.ze_list = ze_list

    def calculate_gram_matrix(self):
        '''self.z_feature.shape : (1, M*64, 768)'''
        M = len(self.zi_list)
        zi_dists, ze_dists = [], []
        for idx in range(M):
            zi_dists.append(self.pairwise_similarities(self.zi_list[idx].clone(), 'rgb'))
            ze_dists.append(self.pairwise_similarities(self.ze_list[idx].clone(), 'event'))
        self.gram_matrix_zi = torch.stack([zi_dists[i].squeeze() for i in range(len(zi_dists))], dim=0)
        self.gram_matrix_ze = torch.stack([ze_dists[i].squeeze() for i in range(len(ze_dists))], dim=0)
        
    def pairwise_similarities(self, z, _type, to_cpu=False):
        T = z.squeeze(0) 
        if _type == 'rgb':
            LIST = torch.cat(self.zi_list, dim=1).squeeze(0) 
        else:
            LIST = torch.cat(self.ze_list, dim=1).squeeze(0)
        
        sim = torch.empty((LIST.size(0) // T.size(0)))  # (M*N, N)
        mean_T = torch.mean(T, dim=0)
        std_T = torch.std(T, dim=0)
        
        for i in range(0, LIST.size(0), T.size(0)):
            element = LIST[i : i+T.size(0)]
            mean_element = torch.mean(element, dim=0)
            std_element = torch.std(element, dim=0)
            cov = torch.sum((T - mean_T) * (element - mean_element), dim=0)
            pearson_coeff = cov / (std_T * std_element)
            sim[i // T.size(0)] = torch.mean(pearson_coeff)
        return sim

    def get_dynamic_z(self, frame_num=1, resample_flag=False):
        if resample_flag:
            random_indices = random.sample(range(len(self.zi_list)), frame_num)
            self.selected_i = [self.zi_list[i] for i in random_indices]
            self.selected_e = [self.ze_list[i] for i in random_indices]
        zi = torch.cat(self.selected_i, dim=0)  # [] -> (M, 64, 768)
        ze = torch.cat(self.selected_e, dim=0)  # [] -> (M, 64, 768)
        #######################################################
        # memory
        if self.net.memory is not None:
            zi, ze = self.net.memory.forward_dynamic_features(zi, ze, None)
        #######################################################
        return zi, ze


class ST_Module(TemplateModule):
    def __init__(self, net, z_capacity, zs=64):
        super(ST_Module, self).__init__(net, z_capacity, zs)
    
    def normed_div_measure(self, matrix_zi, matrix_ze):
        """calculate the normed diversity measure of matirx, the higher the more diverse."""
        assert matrix_zi.shape[0] == matrix_zi.shape[1] and matrix_ze.shape[0] == matrix_ze.shape[1], "Input tensor must be square"
        dim = matrix_zi.shape[0] - 1
        triu_num_zi = int(dim * (dim + 1) / 2)  
        triu_num_ze = int(dim * (dim + 1) / 2) 
        triu_sum_zi = torch.sum(torch.triu(matrix_zi, 1))
        triu_sum_ze = torch.sum(torch.triu(matrix_ze, 1))
        measure_zi = triu_sum_zi / (matrix_zi[0, 0] * triu_num_zi)
        measure_ze = triu_sum_ze / (matrix_ze[0, 0] * triu_num_ze) 
        return 0.5 * measure_zi + 0.5 * measure_ze

    def update_gram_matrix(self, curr_zi, curr_ze):
        # zi/ze.shape (B, 64, 768)
        curr_sims_zi = self.pairwise_similarities(curr_zi, 'rgb').unsqueeze(1) 
        curr_sims_ze = self.pairwise_similarities(curr_ze, 'event').unsqueeze(1) 

        sub_gram_matrix_zi = self.gram_matrix_zi[1:, 1:]
        sub_gram_matrix_zi = torch.cat([sub_gram_matrix_zi, curr_sims_zi[:-1]], dim=1)
        sub_gram_matrix_zi = torch.cat([sub_gram_matrix_zi, curr_sims_zi.T], dim=0)
        self.gram_matrix_zi = sub_gram_matrix_zi

        sub_gram_matrix_ze = self.gram_matrix_ze[1:, 1:]
        sub_gram_matrix_ze = torch.cat([sub_gram_matrix_ze, curr_sims_ze[:-1]], dim=1)
        sub_gram_matrix_ze = torch.cat([sub_gram_matrix_ze, curr_sims_ze.T], dim=0)
        self.gram_matrix_ze = sub_gram_matrix_ze

    def update_z_feature(self, curr_zi, curr_ze):
        self.zi_list.append(curr_zi)
        self.ze_list.append(curr_ze)
        self.zi_list.pop(0)
        self.ze_list.pop(0)
    
    def update(self, zi, ze, curr_zi, curr_ze):
        """append to the current memory and rebuild canvas, return div_scale (diversity of the current memory)
        update distance matrix and calculate the div scale.""" 
        self.zi_raw_list.append(zi)
        self.ze_raw_list.append(ze)
        self.zi_raw_list.pop(0)
        self.ze_raw_list.pop(0)

        self.update_z_feature(curr_zi, curr_ze)
        self.update_gram_matrix(curr_zi, curr_ze)
        div_scale = self.normed_div_measure(matrix_zi=self.gram_matrix_zi, matrix_ze=self.gram_matrix_ze)
        return div_scale


class LT_Module(TemplateModule):
    def __init__(self, net, z_capacity, lower_bound=0.883665, lower_bound_type='dynamic', zs=64):
        super(LT_Module, self).__init__(net, z_capacity, zs)
        self.lower_bound = lower_bound  # lb : 0.90, 0.75, 0.60, 0.45
        self.lower_bound_type = lower_bound_type
        self.filled_idx = 0
    
    def clear(self):
        self.fill(self.static_i, self.static_e)

    def throwaway_or_keep(self, curr_sims_zi, curr_sims_ze, self_sim_zi, self_sim_ze, div_scale):
        """
        determine if we keep the template or not
        if the template is rejected: return -1 (not better) or -2 (rejected by lower bound)
        if we keep the template: return idx where to switch
        """
        base_sim_zi = self.base_sim_zi  
        base_sim_ze = self.base_sim_ze 
        curr_sims_zi = curr_sims_zi.unsqueeze(1) 
        curr_sims_ze = curr_sims_ze.unsqueeze(1) 

        # normalize the gram_matrix, otherwise determinants are huge
        gram_matrix_norm_zi = self.gram_matrix_zi / base_sim_zi 
        gram_matrix_norm_ze = self.gram_matrix_ze / base_sim_ze  
        curr_sims_norm_zi = curr_sims_zi / base_sim_zi 
        curr_sims_norm_ze = curr_sims_ze / base_sim_ze  

        # check if distance to base template is below lower bound
        if self.lower_bound_type == 'dynamic':
            lower_bound = self.lower_bound - (1 - div_scale)  # 0.35-(1-0.95)=0.3, 0.35-(1-0.8)=0.15
            lower_bound = torch.clamp(lower_bound, 0.0, 1.0)  # 

            double_curr_sim = 0.5 * curr_sims_zi[0] + 0.5 * curr_sims_ze[0] 
            double_base_sim = 0.5 * base_sim_zi + 0.5 * base_sim_ze  

            reject = (double_curr_sim < lower_bound * double_base_sim)  
            # print('Reject: ', reject, 'curr_sims: ', curr_sims[0].item(), 
            #       'lb: ', lb.item(), 'base_sim: ', base_sim.item(), 'lb * base_sim: ', (lb * base_sim).item())
        else:
            raise TypeError(f"lower boundary type {self.lower_bound_type} not known.")
        if reject:
            return -2

        # fill the memory with adjacent frames if they are not. populated with something different than the base frame yet
        if self.filled_idx < (self.z_capacity - 1): # (N):0-5, 0:static template，1-N:fill
            self.filled_idx += 1
            throwaway_idx = self.filled_idx
        # determine if and in which spot the template increases the current gram determinant
        else:
            curr_det_zi = torch.det(gram_matrix_norm_zi) 
            curr_det_ze = torch.det(gram_matrix_norm_ze) 

            # start at 1 so we never throwaway the base template
            dets_zi = torch.zeros((self.z_capacity - 1,))  
            dets_ze = torch.zeros((self.z_capacity - 1,))  
    
            # for - (1-self.z_capacity)
            for i in range(self.z_capacity - 1):
                mat_zi = gram_matrix_norm_zi.clone()
                mat_zi[i+1, :] = curr_sims_norm_zi.T
                mat_zi[:, i+1] = curr_sims_norm_zi.T
                mat_zi[i+1, i+1] = self_sim_zi / base_sim_zi
                dets_zi[i] = torch.det(mat_zi)

                mat_ze = gram_matrix_norm_ze.clone()
                mat_ze[i+1, :] = curr_sims_norm_ze.T
                mat_ze[:, i+1] = curr_sims_norm_ze.T
                mat_ze[i+1, i+1] = self_sim_ze / base_sim_ze
                dets_ze[i] = torch.det(mat_ze)

            # check if any of the new combinations is better than the prev. gram_matrix

            norm_curr_det_zi = curr_det_zi / torch.max(dets_zi)
            norm_curr_det_ze = curr_det_ze / torch.max(dets_ze)
            norm_dets_zi = dets_zi / torch.max(dets_zi)
            norm_dets_ze = dets_ze / torch.max(dets_ze)

            weighted_curr_det = 0.5 * norm_curr_det_zi + 0.5 * norm_curr_det_ze 
            weighted_dets = 0.5 * norm_dets_zi + 0.5 * norm_dets_ze

            weighted_max_idx = torch.argmax(weighted_dets)
            if weighted_curr_det > weighted_dets[weighted_max_idx]:
                throwaway_idx = -1
            else:
                throwaway_idx = weighted_max_idx + 1

        assert throwaway_idx != 0
        return throwaway_idx if throwaway_idx != self.z_capacity else -1

    def update_gram_matrix(self, curr_sims_zi, curr_sims_ze, self_sim_zi, self_sim_ze, idx):
        """update the current gram matrix."""
        curr_sims_zi = curr_sims_zi.unsqueeze(1)  
        curr_sims_ze = curr_sims_ze.unsqueeze(1)  
        # add the self similarity at throwaway_idx spot
        curr_sims_zi[idx] = self_sim_zi
        curr_sims_ze[idx] = self_sim_ze

        self.gram_matrix_zi[idx, :] = curr_sims_zi.T
        self.gram_matrix_zi[:, idx] = curr_sims_zi.T
        self.gram_matrix_ze[idx, :] = curr_sims_ze.T
        self.gram_matrix_ze[:, idx] = curr_sims_ze.T

    def update_z_feature(self, curr_zi, curr_ze, idx):
        """update the template feature."""
        self.zi_list[idx] = curr_zi
        self.ze_list[idx] = curr_ze

    def update(self, zi, ze, curr_zi, curr_ze, div_scale, time):
        """decide if the templates is taken into the lt module."""
        curr_sims_zi = self.pairwise_similarities(curr_zi, _type='rgb')
        curr_sims_ze = self.pairwise_similarities(curr_ze, _type='event')

        curr_zi_tmp, curr_ze_tmp = curr_zi.clone(), curr_ze.clone()  
        B, N, C = curr_zi_tmp.size()
        H = W = int(math.sqrt(N))
        curr_zi_tmp, curr_ze_tmp = curr_zi_tmp.reshape(1, C, H, W), curr_ze_tmp.reshape(1, C, H, W) 
        self_sim_zi = F.conv2d(curr_zi_tmp, curr_zi_tmp).squeeze().item()  
        self_sim_ze = F.conv2d(curr_ze_tmp, curr_ze_tmp).squeeze().item()  
        throwaway_idx = self.throwaway_or_keep(curr_sims_zi=curr_sims_zi, curr_sims_ze=curr_sims_ze,
                                                self_sim_zi=self_sim_zi, self_sim_ze=self_sim_ze,
                                                div_scale=div_scale)

        # if the idx is -2 or -1, the template is rejected, otherwise we update
        if throwaway_idx == -2 or throwaway_idx == -1:  # rejected
            # print('div_scale: {}, reject: {}'.format(div_scale, 'True'))
            pass 
            return False
        else:
            # print('div_scale: {}, reject: {}'.format(div_scale, 'False'))
            self.zi_raw_list[throwaway_idx], self.ze_raw_list[throwaway_idx] = zi, ze
            self.timestamp_list[throwaway_idx] = time

            self.update_gram_matrix(curr_sims_zi=curr_sims_zi, curr_sims_ze=curr_sims_ze,
                                    self_sim_zi=self_sim_zi, self_sim_ze=self_sim_ze,
                                    idx=throwaway_idx)
            self.update_z_feature(curr_zi, curr_ze, idx=throwaway_idx)
            return True