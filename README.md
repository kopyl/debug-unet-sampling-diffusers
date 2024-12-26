### Issue:

When training a Unet with 2 channels, images are not generating when calling log_calidation function.
I'm doing the training with this [original script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) but with a few [exceptions](https://github.com/kopyl/debug-unet-sampling-diffusers/pull/1/commits/e8c3fbf359924d460aa1d304b0daa3320ffb0a75):

1. Change amount of input and output channels in Unet to 2.
Replaced
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
to change channels count.

2. Change amount of input and output channels in VAE to 1 and the the latent channels to 2.
Replaced
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