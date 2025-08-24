import torch.nn as nn
from lib.models.layers.hflayers import HopfieldLayer

    
class ATU(nn.Module):
    def __init__(self, template_num=4, embed_dim=768, drop_rate=0.0, memory_dict: dict=None):
        super().__init__()
        self.memory_dict = memory_dict
        self.atu_beta = self.memory_dict['ATU_BETA']
        self.embed_dim = embed_dim
        self.zs = 64
        self.static_z_num = 1
        self.dynamic_z_num = template_num - self.static_z_num 
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.atu_rgb = HopfieldLayer(
            input_size=embed_dim,    
            hidden_size=embed_dim, 
            output_size=embed_dim,
            pattern_size=embed_dim,  
            quantity=1000,  
            stored_pattern_as_static=True,
            state_pattern_as_static=True,
            scaling=self.atu_beta, 
            dropout=0.1,
        )                
        self.atu_event = HopfieldLayer(
            input_size=embed_dim,    
            hidden_size=embed_dim, 
            output_size=embed_dim,
            pattern_size=embed_dim,  
            quantity=1000, 
            stored_pattern_as_static=True,
            state_pattern_as_static=True,
            scaling=self.atu_beta, 
            dropout=0.1,
        )
    
    def forward_dynamic_features(self, dynamic_zi, dynamic_ze, static_zi=None, static_ze=None):
        '''hopfield layer'''
        B, M = dynamic_zi.size(0), dynamic_zi.size(1) // self.zs
        dynamic_zi = dynamic_zi.reshape(-1, self.zs, self.embed_dim)
        dynamic_ze = dynamic_ze.reshape(-1, self.zs, self.embed_dim)
        dynamic_zi = self.atu_rgb(dynamic_zi) + dynamic_zi
        dynamic_ze = self.atu_event(dynamic_ze) + dynamic_ze
        dynamic_zi = dynamic_zi.reshape(B, M*self.zs, self.embed_dim)
        dynamic_ze = dynamic_ze.reshape(B, M*self.zs, self.embed_dim)
        return dynamic_zi, dynamic_ze

def build_atu(cfg, embed_dim):
    template_num = cfg.DATA.TEMPLATE.NUMBER
    memory_dict = cfg.MODEL.MEMORY
    return ATU(template_num=template_num, embed_dim=embed_dim, memory_dict=memory_dict)