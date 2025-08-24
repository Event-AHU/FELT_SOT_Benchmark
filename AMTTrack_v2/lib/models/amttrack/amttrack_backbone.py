import torch
from .vit import VisionTransformer
from lib.models.layers.hflayers import Hopfield


class AMTTrackBackBone(VisionTransformer):
    def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0., asymmetric_flag=False):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, drop_path_rate=drop_path_rate, asymmetric_flag=asymmetric_flag)
        self.embed_dim = embed_dim  # 768
        self.zs = 64
        self.xs = 256

    def finetune_track(self, cfg, patch_start_index=1):
        super().finetune_track(cfg, patch_start_index)
        self._init_hopfield(cfg.MODEL.MEMORY)   
    
    def _init_hopfield(self, memory_dict):
        self.memory_dict = memory_dict
        self.amah_beta = self.memory_dict['AMAH_BETA']
        self.amah_layers = self.memory_dict['AMAH_LAYERS']
        self.amah_sp_layers = self.memory_dict['AMAH_SP_LAYERS']
        self.amah_rgb = Hopfield(
            input_size=self.embed_dim,    
            hidden_size=self.embed_dim,
            output_size=self.embed_dim,
            pattern_size=self.embed_dim,   
            scaling=self.amah_beta,  
            dropout=0.1,
        )
        self.amah_event = Hopfield(
            input_size=self.embed_dim,    
            hidden_size=self.embed_dim,
            output_size=self.embed_dim,
            pattern_size=self.embed_dim,   
            scaling=self.amah_beta,  
            dropout=0.1,
        )
    
    def _z_feat(self, zi):
        B, M, C, H, W = zi.size()  
        zi = zi.reshape(-1, C, H, W)
        zi = self.patch_embed(zi)  
        zi += self.pos_embed_z  
        zi = zi.reshape(B, M*self.zs, self.embed_dim)
        return zi

    def _x_feat(self, xi):
        xi = self.patch_embed(xi)  
        xi += self.pos_embed_x  
        return xi
    
    def _amah(self, x, i, lens_z):
        if self.amah_sp_idx < len(self.amah_sp_layers) and i in self.amah_sp_layers[self.amah_sp_idx]:
            self.amah_sp_dict.update({f'stored_patterns_{i}_rgb': x[:, lens_z:lens_z+self.xs, :]})
            self.amah_sp_dict.update({f'stored_patterns_{i}_event': x[:, lens_z+self.xs:, :]})
            if i == self.amah_sp_layers[self.amah_sp_idx][-1]:
                self.amah_sp_idx += 1
        if i in self.amah_layers:
            R_x_rgb = x[:, lens_z:lens_z+self.xs, :]
            R_x_event = x[:, lens_z+self.xs:, :]
            hop_rgb = torch.cat([self.amah_sp_dict[f'stored_patterns_{k}_rgb'] for k in self.amah_sp_layers[self.amah_hop_idx]], dim=1)
            hop_event = torch.cat([self.amah_sp_dict[f'stored_patterns_{k}_event'] for k in self.amah_sp_layers[self.amah_hop_idx]], dim=1)
            self.amah_hop_idx += 1
            R_x_rgb = self.amah_event((hop_event, R_x_rgb, hop_event)) + R_x_rgb
            R_x_event = self.amah_rgb((hop_rgb, R_x_event, hop_rgb)) + R_x_event
            if i != 11:
                self.amah_sp_dict.update({f'stored_patterns_{i}_rgb': R_x_rgb})  
                self.amah_sp_dict.update({f'stored_patterns_{i}_event': R_x_event})  
            x = torch.cat([x[:, :lens_z, :], R_x_rgb, R_x_event], dim=1)
        return x
    
    def forward(self, static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe,
                mask_z=None, mask_x=None, ce_template_mask=None, ce_keep_rate=None, 
                return_last_attn=False):

        lens_z = static_zi.size(1) + static_ze.size(1) + dynamic_zi.size(1) + dynamic_ze.size(1)
        lens_x = xi.size(1) + xe.size(1)
        x = torch.cat((static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe), dim=1)
        x = self.pos_drop(x)

        self.amah_sp_dict = {}
        self.amah_sp_idx = 0
        self.amah_hop_idx = 0
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, lens_z=lens_z, lens_x=lens_x)
            x = self._amah(x, i, lens_z)
        x = self.norm(x)  

        aux_dict = {"attn": attn,}
        return x, aux_dict


def _create_amttrack_vit(pretrained=False, **kwargs):
    model = AMTTrackBackBone(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)
            print('Missing keys:', missing_keys)
            print('Unexpected keys:', unexpected_keys)
    return model

def amttrack_vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_amttrack_vit(pretrained=pretrained, **model_kwargs)
    return model