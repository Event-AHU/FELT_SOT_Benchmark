import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Coesot, Fe108, VisEvent, Felt
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["COESOT", "COESOT_VAL", "FE108", "FE108_VAL", "VisEvent", "VisEvent_VAL", "FELT", "FELT_VAL", 
                        "COESOT_TEST", "FELT_TEST", "FE108_TEST"]
        if name == "COESOT":
            datasets.append(Coesot(settings.env.coesot_dir, split='train', image_loader=image_loader))
        if name == "COESOT_VAL":
            datasets.append(Coesot(settings.env.coesot_val_dir, split='val', image_loader=image_loader))
        if name == "FE108":
            datasets.append(Fe108(settings.env.fe108_dir, split='train', image_loader=image_loader))
        if name == "FE108_VAL":
            datasets.append(Fe108(settings.env.fe108_val_dir, split='val', image_loader=image_loader))
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir, split='train', image_loader=image_loader))
        if name == "VisEvent_VAL":
            datasets.append(VisEvent(settings.env.visevent_val_dir, split='val', image_loader=image_loader))
        if name == "FELT":
            datasets.append(Felt(settings.env.felt_dir, split='train', image_loader=image_loader))
        if name == "FELT_VAL":
            datasets.append(Felt(settings.env.felt_val_dir, split='val', image_loader=image_loader))

    return datasets


def build_dataloaders(cfg, settings):
    if cfg.DATA.TRAIN.DATASETS_NAME[0] == "FE108":
        transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.5))  # for FE108 p=0.5 else 0.05
        transform_train = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
        transform_val = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    else:
        transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(0.5))
        transform_train = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD), tfm.RandomHorizontalFlip_Norm(0.5))  # for FE108 not normalize else normalize  
        transform_val = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    data_processing_train = processing.STARKProcessing(search_area_factor=settings.search_area_factor, 
                                                       output_sz=settings.output_sz, 
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,  
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=settings.search_area_factor, 
                                                     output_sz=settings.output_sz,  
                                                     center_jitter_factor=settings.center_jitter_factor,  
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)  
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)  
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal") 
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)

    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO, 
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,   
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,  
                                            num_search_frames=settings.num_search, num_template_frames=settings.num_template, 
                                            processing=data_processing_train, 
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)  

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None  # None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, 
                                          num_search_frames=settings.num_search, num_template_frames=settings.num_template, 
                                          processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)

    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None  # None

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    pretrain_file = getattr(cfg.MODEL, "PRETRAIN_FILE", "None")
    
    ######################################################################################################
    if 'mae' in pretrain_file:
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR,
            },
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and "amah" not in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in net.named_parameters() if "amah" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in net.named_parameters() if "atu" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters(): 
                if p.requires_grad:
                    print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)
    ######################################################################################################
    elif 'OSTrack' in pretrain_file:
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR  * cfg.TRAIN.BACKBONE_MULTIPLIER  * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and "amah" not in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER  * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in net.named_parameters() if "amah" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in net.named_parameters() if "atu" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters(): 
                if p.requires_grad:
                    print(n, p.numel())

        total_num = sum(p.numel() for n, p in net.named_parameters())
        trainable_num = sum(p.numel() for n, p in net.named_parameters() if p.requires_grad)
        print('=Total: ',total_num, 'Trainable: ',trainable_num)
    ######################################################################################################
    else:
        raise ValueError("Unsupported pretrain_file")
    ######################################################################################################

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler