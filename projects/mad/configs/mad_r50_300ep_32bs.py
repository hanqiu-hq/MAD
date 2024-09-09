from fvcore.common.param_scheduler import MultiStepParamScheduler

from detrex.config import get_config
from detectron2.solver import WarmupParamScheduler
from detectron2.config import LazyCall as L


dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./outputs/mad_300ep"

# max training iterations
train.max_iter = 1125000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[750000, 975000],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# run evaluation every 7500 iters
train.eval_period = 7500

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 3750 iters
train.checkpointer.period = 3750
train.checkpointer.max_to_keep = 3

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

dataloader.train.num_workers = 8
dataloader.train.total_batch_size = 32
dataloader.evaluator.output_dir = train.output_dir