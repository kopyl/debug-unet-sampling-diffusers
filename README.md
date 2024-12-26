### Issue description:

When doing `examples/text_to_image/train_text_to_image.py` training a Unet with 2 channels, images are not generating when calling `log_validation` function.

It gives me this error:
```
setStorage: sizes [320, 1280], strides [1, 320], storage offset 6080, and itemsize 2 requiring a storage size of 831360 are out of bounds for storage of size 0
```
Detailed logs are added [at the end of this README](https://github.com/kopyl/debug-unet-sampling-diffusers/tree/main?tab=readme-ov-file#detailed-logs).

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
and got different error:
```
size of input tensor and input format are different. tensor shape: (1, 512, 512), input_format: NHWC
```
Here is the detailed log.