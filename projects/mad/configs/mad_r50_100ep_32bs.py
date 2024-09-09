from detrex.config import get_config

dataloader = get_config("common/data/coco_detr.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./outputs/mad_100ep"

# max training iterations
train.max_iter = 375000

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