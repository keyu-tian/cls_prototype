from ema import EMA
from meta import NUM_CLASSES
from models.effnet import efficientnet_b3


def build_model(model_cfg):
    name, kw = model_cfg.name, model_cfg.kwargs
    kw.update({'num_classes': NUM_CLASSES})
    model = {
        'efficientnet_b3': efficientnet_b3
    }[name](**kw).cuda()
    ema = EMA(model, model_cfg.ema_mom)
    return ema, model
