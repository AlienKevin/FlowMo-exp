"""FlowMo train script."""

import glob
import os
import shutil
import time
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fsspec
import lpips
import torch
import torch.distributed as dist
import torch.optim as optim
from mup import MuAdam, MuAdamW
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import deepspeed

from flowmo import models, perceptual_loss, train_utils

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

BFLOAT16_IS_AVAILABLE = None


def _get_norm(model, getter):
    return sum(
        (getter(p) ** 2).sum() for p in model.parameters() if p.grad is not None
    ).sqrt()


def train_step(config, model, batch, optimizer, aux_state):
    assert BFLOAT16_IS_AVAILABLE is not None
    dtype = torch.bfloat16 if BFLOAT16_IS_AVAILABLE else torch.float32

    aux = {"loss_dict": {}}
    model.zero_grad()

    with torch.autocast(
        "cuda",
        dtype=dtype,
    ):
        loss, aux = models.rf_loss(config, model, batch, aux_state)

    model.backward(loss)

    if config.opt.log_norms:
        pass
        # TODO
        # clipped_grad_norm = _get_norm(model.module, getter=lambda p: p.grad)
        # aux["loss_dict"]["debug/clipped_grad_norm"] = clipped_grad_norm
        # aux["loss_dict"]["debug/param_norm"] = _get_norm(model.module, getter=lambda p: p)

    model.step()
    return loss.detach(), aux


def main(args, config):
    config = train_utils.restore_config(config)
    print(torch.__version__)
    models.MUP_ENABLED = config.model.enable_mup

    # train_utils.soft_init()
    deepspeed.init_distributed()

    rank = dist.get_rank()
    print(rank)
    dist.barrier()

    log_dir = os.path.join(args.results_dir, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        '/', 'tmp', 'kevin02', f"{args.experiment_name}_torchinductor_cache_{str(rank)}"
    )

    device = rank % torch.cuda.device_count()
    print(device, torch.cuda.device_count())

    # torch.cuda.set_device(device) # Deepspeed handles this

    global BFLOAT16_IS_AVAILABLE
    BFLOAT16_IS_AVAILABLE = (
        train_utils.bfloat16_is_available() and config.trainer.enable_bfloat16
    )
    print("Using bfloat16: ", BFLOAT16_IS_AVAILABLE)

    torch.manual_seed(0)

    model = train_utils.build_model(config)

    # if BFLOAT16_IS_AVAILABLE:
    #     model = model.to(torch.bfloat16)

    aux_state = {}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params: {n_params}")

    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    # model = DistributedDataParallel(model, find_unused_parameters=True)

    if config.model.enable_mup:
        if config.opt.weight_decay:
            opt_cls = MuAdamW
        else:
            opt_cls = MuAdam
    else:
        if config.opt.weight_decay:
            opt_cls = optim.AdamW
        else:
            opt_cls = optim.Adam

    encoder_pg = {
        "params": [p for (n, p) in model.named_parameters() if "encoder" in n],
        "lr": config.opt.lr,
    }
    decoder_pg = {
        "params": [p for (n, p) in model.named_parameters() if "decoder" in n],
        "lr": config.opt.lr,
    }
    prior_pg = {
        "params": [p for (n, p) in model.named_parameters() if "prior_model" in n],
        "lr": config.opt.lr * config.prior.lr_multiplier,
    }
    quantizer_pg = {
        "params": [p for (n, p) in model.named_parameters() if ("quantizer" in n and "prior_model" not in n)],
        "lr": config.opt.lr,
    }
    all_params = set(p for n, p in model.named_parameters())
    assert set(encoder_pg["params"]).union(set(decoder_pg["params"])).union(set(prior_pg["params"])).union(set(quantizer_pg["params"])) == all_params

    def build_optimizer(pgs):
        optimizer = opt_cls(
            pgs,
            weight_decay=config.opt.weight_decay,
            betas=(config.opt.beta1, config.opt.beta2),
        )
        return optimizer

    optimizer = build_optimizer([encoder_pg, decoder_pg, prior_pg, quantizer_pg])
    
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    # Inject the batch size and accumulation steps from our main config
    ds_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    ds_config['gradient_accumulation_steps'] = config.opt.n_grad_acc
    
    # We pass the config dict directly to initialize,
    # and we nullify the config path in args to prevent
    # deepspeed from trying to load it again.
    args.deepspeed_config = None

    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    rebuilt_optimizer = False

    train_dataloader = train_utils.load_dataset(config, split='train')

    if rank == 0:
        writer = SummaryWriter(log_dir)

    total_steps = 0

    load_dir = os.path.join(log_dir, "checkpoints")
    if args.resume_from_ckpt:
        # tag is the checkpoint_id. resume_from_ckpt is the full path to the checkpoint folder
        tag = os.path.basename(args.resume_from_ckpt)
        _, client_state = model.load_checkpoint(args.resume_from_ckpt, tag=tag)
        total_steps = client_state['total_steps']
    else:
        # Try to find latest checkpoint in the default directory
        _, client_state = model.load_checkpoint(load_dir)
        if client_state:
            total_steps = client_state['total_steps']

    # model_ema = train_utils.SimpleEMA(model.module, decay=config.model.ema_decay)

    tic = time.time()
    dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

    print("Training begins.")
    print(args)
    print(OmegaConf.to_yaml(config))
    if rank == 0:
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))

    with torch.no_grad():
        if config.model.fix_initial_norms:
            if config.model.fix_norm_mode == "channel":
                norm_kwargs = dict(axis=1, keepdims=True)
            elif config.model.fix_norm_mode == "l2":
                norm_kwargs = dict()
            else:
                raise NotImplementedError

            initial_norms = {
                k: torch.linalg.norm(v, **norm_kwargs)
                for (k, v) in models.get_weights_to_fix(model)
            }
            print("Norms checksum", sum(v.sum() for v in initial_norms.values()))

    if config.opt.lpips_weight != 0.0:
        if config.opt.lpips_mode == "vgg":
            aux_state["lpips_model"] = (
                lpips.LPIPS(net="vgg").eval().requires_grad_(False).cuda()
            )
        elif config.opt.lpips_mode == "resnet":
            aux_state["lpips_model"] = (
                perceptual_loss.PerceptualLoss().eval().requires_grad_(False).cuda()
            )
        else:
            raise NotImplementedError

    running_losses = {}

    aux_state["dl_iter"] = dl_iter

    while total_steps <= config.trainer.max_steps:
        model.train()
        if config.opt.freeze_encoder or total_steps >= config.opt.freeze_encoder_after:
            if not rebuilt_optimizer:
                print(f"Rebuilding optimizer at step {total_steps}")
                
                # Freeze all parameters in param group 0 and 3 (encoder and quantizer)
                for param in [param for i in [0, 3] for param in optimizer.param_groups[i]['params']]:
                    param.requires_grad = False
                optimizer.param_groups[0]['lr'] = 0
                optimizer.param_groups[3]['lr'] = 3
                
                # Update prior LR
                optimizer.param_groups[2]['lr'] = config.opt.lr
                model.module.config.prior.loss_weight = 1.0

                # model.module.config.prior.stop_grad = True
                # optimizer = build_optimizer([decoder_pg, prior_pg])
                rebuilt_optimizer = True
                # model_ema.decay = config.model.ema_decay

        dl_tic = time.time()
        batch = next(dl_iter)
        dl_toc = time.time()
        if dl_toc - dl_tic > 1.0:
            print(f"Dataloader took {dl_toc - dl_tic} seconds!")
        images = batch["image"]

        aux_state["total_steps"] = total_steps

        loss, aux = train_step(config, model, batch, optimizer, aux_state)
        loss_dict = aux["loss_dict"]

        for k, v in loss_dict.items():
            if k in running_losses:
                running_losses[k] += v
            else:
                running_losses[k] = v

        if config.model.fix_initial_norms:
            for name, weight in models.get_weights_to_fix(model):
                weight.data = (
                    weight
                    / torch.linalg.norm(weight, **norm_kwargs)
                    * initial_norms[name]
                )

        # model_ema.update(model.module, step=total_steps)

        total_steps += 1

        if total_steps == 1:
            print("first step done!")
            print(images.min(), images.max(), images.mean())

        # Refresh dataloader
        if total_steps % 10_000 == 0:
            train_dataloader = train_utils.load_dataset(config, split='train')
            dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

        if total_steps % config.trainer.log_every == 0:
            toc = time.time()
            torch.cuda.synchronize()

            steps_per_sec = config.trainer.log_every / (toc - tic)
            running_losses = {
                k: (l / config.trainer.log_every).item()
                for (k, l) in running_losses.items()
            }
            reserved_gb = torch.cuda.max_memory_reserved() / 1e9
            allocated_gb = torch.cuda.max_memory_allocated() / 1e9

            if rank == 0:
                with torch.no_grad():
                    # For ZeRO-2, we need to gather the parameters to calculate the checksum
                    with deepspeed.zero.GatheredParameters(model.module.encoder.parameters()):
                        encoder_checksum = sum(
                            p.mean() for p in model.module.encoder.parameters()
                        ).item()
                        running_losses["encoder_checksum"] = encoder_checksum

            print(
                dict(
                    memory_usage=train_utils.memory_usage(),
                    total_steps=total_steps,
                    steps_per_sec=steps_per_sec,
                    reserved_gb=reserved_gb,
                    allocated_gb=allocated_gb,
                    **running_losses,
                )
            )

            if rank == 0:
                for k, v in running_losses.items():
                    writer.add_scalar(k, v, global_step=total_steps)
                writer.add_scalar(
                    "Steps per sec", steps_per_sec, global_step=total_steps
                )

            tic = time.time()
            running_losses = dict()

        if total_steps % config.trainer.checkpoint_every == 0:
            checkpoint_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_id = "%.8d" % total_steps

            model.save_checkpoint(checkpoint_dir, checkpoint_id, client_state={"total_steps": total_steps})

            # Remove old checkpoints
            if rank == 0:
                checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*")))
                for checkpoint in checkpoints[:-2]:
                    ckpt_step = int(os.path.basename(checkpoint))
                    if (ckpt_step % config.trainer.keep_every) != 0:
                        shutil.rmtree(checkpoint)

            print("after checkpoint save:")
            print(
                dict(
                    reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                    allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                )
            )


if __name__ == "__main__":
    try:
        args, config = train_utils.get_args_and_config()
        main(args, config)
    finally:
        if dist.is_initialized():
            torch.distributed.destroy_process_group()