### UPD: FIXED

The problem was in the bad accelerate config. I just used the default one and it solved the issue.
Forgot to share the faulty config, but here is the one that fixed the issue:
```
{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "distributed_type": "MULTI_GPU",
  "downcast_bf16": false,
  "enable_cpu_affinity": false,
  "machine_rank": 0,
  "main_training_function": "main",
  "mixed_precision": "no",
  "num_machines": 1,
  "num_processes": 4,
  "rdzv_backend": "static",
  "same_network": false,
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false
}
```

### Issue description:

When doing `examples/text_to_image/train_text_to_image.py` training a Unet with 2 channels, images are not generating when calling `log_validation` function.

The main cause is `accelerate`. Without it 

I get this error:
```
setStorage: sizes [320, 1280], strides [1, 320], storage offset 6080, and itemsize 2 requiring a storage size of 831360 are out of bounds for storage of size 0
```
[Detailed logs are here](https://github.com/kopyl/debug-unet-sampling-diffusers/blob/e8cd7c38c840e801b419789e5832b796337ebc3f/detailed-logs/initial-training-with-accelerate.log).

I [tried debugging it](https://github.com/kopyl/debug-unet-sampling-diffusers/tree/main?tab=readme-ov-file#what-i-tried-to-fix-it), but got no luck yet.

### Setup:

I'm doing the training with [this script](https://github.com/kopyl/debug-unet-sampling-diffusers/blob/dfcec07bdec57abc442bb7bc86134eb91ebe05cc/train_text_to_image.py), which is [the original script](https://github.com/huggingface/diffusers/blob/1b202c5730631417000585e3639539cefc79cbd7/examples/text_to_image/train_text_to_image.py) but with a few [exceptions](https://github.com/kopyl/debug-unet-sampling-diffusers/pull/1/commits/e8c3fbf359924d460aa1d304b0daa3320ffb0a75):

<br />
<br />

1. Changed amount of input and output channels in Unet to 2 by

<br />

replacing

```
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)
```
with
```
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="unet",
    revision=args.non_ema_revision,
    in_channels=2,
    out_channels=2,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
)
```
<br />
<br />

2. Changed amount of input and output channels in VAE to 1 and the the latent channels to 2 by

<br />

replacing

```
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
)
```
with
```
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="vae",
    revision=args.revision,
    variant=args.variant,
    in_channels=1,
    out_channels=1,
    latent_channels=2,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
)
```
<br />
<br />

3. Changed dataset input images conversion from RGB to 1-channel (`"L"`) by

<br />

replacing

```
images = [image.convert("RGB") for image in examples[image_column]]
```
with
```
images = [image.convert("L") for image in examples[image_column]]
```
<br />
<br />

4. Made generating images sooner by

<br />

replacing

```
accelerator.save_state(save_path)
```
with
```
log_validation(
    vae,
    text_encoder,
    tokenizer,
    unet,
    args,
    accelerator,
    weight_dtype,
    global_step,
)
```


### Training launching:
```
accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --dataset_name=reach-vb/pokemon-blip-captions \
  --resolution=256 \
  --train_batch_size=32 \
  --max_train_steps=10000000 \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --noise_offset=0.05 \
  --validation_prompts "photo of a cat" \
  --seed=1 \
  --output_dir="/trained_unet_test" \
  --checkpointing_steps=2 \
  --checkpoints_total_limit=1
```

### What i tried to fix it:

I tried simplifying the training as much as possible.
You can run [this training script](https://github.com/kopyl/debug-unet-sampling-diffusers/blob/6bbb493a905f801942551cb9bcaff3a05910a5e5/train_text_to_image_SIMPLIFIED.py)
<br />
like

```
python train_text_to_image_SIMPLIFIED.py
```

And see that the training and image generation works just fine.

I also tried running the training script without `accelerate` (which is not really an option because i need multi-GPU training)
<br />
with command
```
python train_text_to_image.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --dataset_name=reach-vb/pokemon-blip-captions \
  --resolution=256 \
  --train_batch_size=32 \
  --max_train_steps=10000000 \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --noise_offset=0.05 \
  --validation_prompts "photo of a cat" \
  --seed=1 \
  --output_dir="/trained_unet_test" \
  --checkpointing_steps=2 \
  --checkpoints_total_limit=1
```
and got a different error:
```
size of input tensor and input format are different. tensor shape: (1, 512, 512), input_format: NHWC
```
[Here are the detailed logs](https://github.com/kopyl/debug-unet-sampling-diffusers/blob/7ccf860ce0f736829865f46bb681be8f7c047914/detailed-logs/running-without-accelerate.log).

Which was fixable by adding `--report_to="wandb"` arg to my training args command.
So I added it and launched the training again. This fixed everything.
So this was accelerate that caused this issue, now I need to find a way to run the training with accelerate...

Then I tried [implementing simple text-to-image training script which supports multi-GPU training](https://github.com/kopyl/debug-unet-sampling-diffusers/blob/edbbb8faa5cdce968009d07ceea27bfd300ea842/train_text_to_image_SIMPLIFIED_WITH_MULTI_GPU.py) to localize the issue.
I ran it with
```
accelerate launch --mixed_precision="fp16" train_text_to_image_SIMPLIFIED_WITH_MULTI_GPU.py
```
And the training and image generation went smoothly.

If it tells you something, then in [this simplified script](https://github.com/kopyl/debug-unet-sampling-diffusers/blob/edbbb8faa5cdce968009d07ceea27bfd300ea842/train_text_to_image_SIMPLIFIED_WITH_MULTI_GPU.py) (which generates images perfectly) the `pipeline` [handles time embeddings](https://github.com/huggingface/diffusers/blob/1b202c5730631417000585e3639539cefc79cbd7/src/diffusers/models/unets/unet_2d_condition.py#L1141) in following shaped:
- `t_emb` shape is `[2, 320]` [batch_size?, ?]
- `sample` shape is `[2, 2, 64, 64]` [channels_in?, channels_out?, sample_size?, sample_size?]
- `timestep` shape is `[]` (have no clue why it's empty)
- `sample`'s dtype is `float32`

while in [the original one](https://github.com/huggingface/diffusers/blob/1b202c5730631417000585e3639539cefc79cbd7/examples/text_to_image/train_text_to_image.py) (which fails) it's:
- `t_emb` shape is `[32, 320]` [batch_size?, ?]
- `sample` shape is `[32, 2, 32, 32]` [channels_in?, channels_out?, sample_size?, sample_size?]
- `timestep` shape is `[32]`
- `sample`'s dtype is `float16` (so the difference is in the dtype...)

To make sure i'm not tripping i tried launching [the original training script](https://github.com/huggingface/diffusers/blob/1b202c5730631417000585e3639539cefc79cbd7/examples/text_to_image/train_text_to_image.py) with [JUST ONE MODIFICATION](https://github.com/kopyl/debug-unet-sampling-diffusers/pull/2/commits/a93157deab882f449c560da5cfc216fbe823ce87) – [making the validation sooner](https://github.com/kopyl/debug-unet-sampling-diffusers/blob/main/train_text_to_image_WITH_JUST_LOGGING_ADDED.py#L1073). And still the same issue.

<br />
<br />


### Device info:

- Linux
- VM on Azure with x4 NVIDIA A100
- Python 3.8.10

Packages info:
- accelerate==1.0.1
- transformers==4.46.3
- diffusers-0.33.0.dev0 . Installed with `pip install git+https://github.com/huggingface/diffusers.git` (commit # 1b202c5730631417000585e3639539cefc79cbd7)