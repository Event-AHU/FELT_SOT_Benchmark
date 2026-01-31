import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.layers.atu import build_atu
from lib.models.amttrack.amttrack_backbone import amttrack_vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh


class AMTTrack(nn.Module):
    def __init__(self, transformer, memory, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.memory = memory
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)  
            self.feat_sz_z = int(box_head.feat_sz / 2)  
            self.feat_len_s = int(self.feat_sz_s ** 2)  
            self.feat_len_z = int(self.feat_sz_z ** 2)  

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, 
                zi: torch.Tensor, ze: torch.Tensor, xi: torch.Tensor, xe: torch.Tensor,
                mask_z=None, ce_template_mask=None, ce_keep_rate=None,
                return_last_attn=False,
                ):
        static_zi = zi[:, [0], :, :, :]  # shape (B, 1, C, H, W)
        static_ze = ze[:, [0], :, :, :]  
        dynamic_zi = zi[:, 1:, :, :, :]  # shape (B, M, C, H, W)  
        dynamic_ze = ze[:, 1:, :, :, :]   
        static_zi = self.backbone._z_feat(static_zi)  
        static_ze = self.backbone._z_feat(static_ze)  
        dynamic_zi = self.backbone._z_feat(dynamic_zi)  
        dynamic_ze = self.backbone._z_feat(dynamic_ze)  

        # memory
        if self.memory is not None:
            dynamic_zi, dynamic_ze = self.memory.forward_dynamic_features(dynamic_zi, dynamic_ze)

        xi = xi.squeeze(1)
        xe = xe.squeeze(1)
        xi = self.backbone._x_feat(xi)
        xe = self.backbone._x_feat(xe)

        x, aux_dict = self.backbone(static_zi=static_zi, static_ze=static_ze, 
                                    dynamic_zi=dynamic_zi, dynamic_ze=dynamic_ze,
                                    xi=xi, xe=xe,
                                    mask_z=mask_z, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)  
        out['backbone_feat'] = x
        return out  

    def inference(self, 
                static_zi: torch.Tensor, static_ze: torch.Tensor, 
                dynamic_zi: torch.Tensor, dynamic_ze: torch.Tensor,
                xi: torch.Tensor, xe: torch.Tensor,
                mask_z=None, ce_template_mask=None, ce_keep_rate=None,
                return_last_attn=False,
                ):
        static_zi = self.backbone._z_feat(static_zi.unsqueeze(1))  
        static_ze = self.backbone._z_feat(static_ze.unsqueeze(1))  

        xi = self.backbone._x_feat(xi)
        xe = self.backbone._x_feat(xe)

        x, aux_dict = self.backbone(static_zi=static_zi, static_ze=static_ze,
                                    dynamic_zi=dynamic_zi, dynamic_ze=dynamic_ze,
                                    xi=xi, xe=xe,
                                    mask_z=mask_z, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, 
                                    return_last_attn=return_last_attn)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        out.update(aux_dict)  
        out['backbone_feat'] = x
        return out


    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s*2:]  # extract search region
        enc_opt_x = enc_opt[:, :self.feat_len_s]
        enc_opt_event_x = enc_opt[:, -self.feat_len_s:]  
        enc_opt = enc_opt_x + enc_opt_event_x  
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            '''
            score_map_ctr: (B, 1, 16, 16)
            bbox: (B, 4)
            size_map: (B, 2, 16, 16)
            offset_map: (B, 2, 16, 16)
            '''
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_amttrack(cfg, training=True):
    pretrained_path = cfg.MODEL.PRETRAIN_PATH

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'amttrack_vit_base_patch16_224':
        asymmetric_flag = True if 'mae' in cfg.MODEL.PRETRAIN_FILE else False
        backbone = amttrack_vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, asymmetric_flag=asymmetric_flag,)
        patch_start_index = 1
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim

    # Backbone-finetune
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    # Memory
    memory = build_atu(cfg, backbone.embed_dim)
    # Head
    box_head = build_box_head(cfg, hidden_dim)

    model = AMTTrack(
        backbone,
        memory,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False) 
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)

    return model