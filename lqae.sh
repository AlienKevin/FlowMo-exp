patch_size=8
code_length=1024
mup_width=4
batch_size=16

torchrun -m flowmo.train \
    --experiment-name "flowmo_qwen3-0.6b_pretrain" \
    model.context_dim=768 model.quantization_type="qwen3-0.6b-base" model.code_length=${code_length} \
    model.patch_size=${patch_size} \
    model.mup_width=${mup_width} \
    data.batch_size=${batch_size}\
    trainer.max_steps=100000 \
    trainer.checkpoint_every=5000 \
    trainer.keep_every=5000
