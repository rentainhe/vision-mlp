# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_mlp import SwinMLP
from .res_mlp import ResMLP
from .s2_mlp import S2MLP


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == "res_mlp":
        model = ResMLP(img_size=config.DATA.IMG_SIZE,
                       patch_size=config.MODEL.RESMLP.PATCH_SIZE,
                       num_classes=config.MODEL.NUM_CLASSES,
                       in_chans=config.MODEL.RESMLP.IN_CHANS,
                       embed_dim=config.MODEL.RESMLP.EMBED_DIM,
                       depth=config.MODEL.RESMLP.DEPTH,
                       drop_rate=config.MODEL.DROP_RATE,
                       drop_path_rate=config.MODEL.DROP_PATH_RATE,
                       init_scale=config.MODEL.RESMLP.INIT_SCALE)
    elif model_type == "s2_mlp":
        model = S2MLP(img_size=config.DATA.IMG_SIZE,
                      patch_size=config.MODEL.S2MLP.PATCH_SIZE,
                      in_chans=config.MODEL.S2MLP.IN_CHANS,
                      embed_dim=config.MODEL.S2MLP.EMBED_DIM,
                      depth=config.MODEL.S2MLP.DEPTH,
                      expand_ratio=config.MODEL.S2MLP.EXPAND_RATIO,
                      mlp_bias=config.MODEL.S2MLP.MLP_BIAS,
                      num_classes=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
