from transformers import AdamW
import functools
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    1. D_mul：过度参数化的深度乘法器。注意，groups参数在DO-Conv (groups=1)，DO-DConv (groups=in_channels), DO-GConv（否则）。
    一段预热期，在此期间，从0到优化器中的初始lr设置线性增加。

    参数:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    返回:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(optimizer, functools.partial(lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps), last_epoch)


def config_optimizer(parameters, init_lr, warmup_steps, max_steps, name='lr'):
    """
    原来的伯特优化器不衰减偏差和layer_normal
    参数:
        parameters:
        init_lr:
        warmup_steps:
        max_steps:
        name:
        weight_decay:

    返回:

    """
    optimizer = AdamW(
        parameters, lr=init_lr, eps=1e-8, correct_bias=False
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )
    scheduler = {'scheduler': scheduler, 'name': name, 'interval': 'step', 'frequency': 1}

    return optimizer, scheduler