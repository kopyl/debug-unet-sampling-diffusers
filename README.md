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


### Detailed logs:

```
Steps:   0%|                                                                                                                           | 0/10000000 [00:00<?, ?it/s]/usr/local/lib/python3.8/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
/usr/local/lib/python3.8/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
/usr/local/lib/python3.8/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
/usr/local/lib/python3.8/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
Steps:   0%|                                                                                       | 2/10000000 [00:12<15309:10:48,  5.51s/it, lr=1e-5, step_loss=1]12/26/2024 02:11:11 - INFO - __main__ - 1 checkpoints already exist, removing 1 checkpoints
12/26/2024 02:11:11 - INFO - __main__ - removing checkpoints: checkpoint-2
12/26/2024 02:11:11 - INFO - __main__ - Running validation...
[2024-12-26 02:11:11,173] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
 [WARNING]  On Ampere and higher architectures please use CUDA 11+
 [WARNING]  On Ampere and higher architectures please use CUDA 11+
 [WARNING]  On Ampere and higher architectures please use CUDA 11+
 [WARNING]  On Ampere and higher architectures please use CUDA 11+
 [WARNING]  On Ampere and higher architectures please use CUDA 11+
 [WARNING]  On Ampere and higher architectures please use CUDA 11+
{'image_encoder', 'requires_safety_checker'} was not found in config. Values will be initialized to default values.
                                                                                                                                                                   Loaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of stable-diffusion-v1-5/stable-diffusion-v1-5.              | 0/6 [00:00<?, ?it/s]
{'prediction_type', 'timestep_spacing'} was not found in config. Values will be initialized to default values.
Loaded scheduler as PNDMScheduler from `scheduler` subfolder of stable-diffusion-v1-5/stable-diffusion-v1-5.
Loading pipeline components...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 882.33it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
[rank0]: Traceback (most recent call last):
[rank0]:   File "train-original-script.py", line 1176, in <module>
[rank0]:     main()
[rank0]:   File "train-original-script.py", line 1088, in main
[rank0]:     log_validation(
[rank0]:   File "train-original-script.py", line 174, in log_validation
[rank0]:     image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py", line 1008, in __call__
[rank0]:     noise_pred = self.unet(
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/accelerate/utils/operations.py", line 820, in forward
[rank0]:     return model_forward(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/accelerate/utils/operations.py", line 808, in __call__
[rank0]:     return convert_to_fp32(self.model_forward(*args, **kwargs))
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/amp/autocast_mode.py", line 43, in decorate_autocast
[rank0]:     return func(*args, **kwargs)
[rank0]:   File "/diffusers/src/diffusers/models/unets/unet_2d_condition.py", line 1143, in forward
[rank0]:     emb = self.time_embedding(t_emb, timestep_cond)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/diffusers/src/diffusers/models/embeddings.py", line 753, in forward
[rank0]:     sample = self.linear_1(sample)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/linear.py", line 117, in forward
[rank0]:     return F.linear(input, self.weight, self.bias)
[rank0]: RuntimeError: setStorage: sizes [320, 1280], strides [1, 320], storage offset 6080, and itemsize 2 requiring a storage size of 831360 are out of bounds for storage of size 0
Steps:   0%|                                                                                       | 2/10000000 [00:15<21117:40:27,  7.60s/it, lr=1e-5, step_loss=1]
W1226 02:11:16.154098 140385973630784 torch/distributed/elastic/multiprocessing/api.py:858] Sending process 937053 closing signal SIGTERM
W1226 02:11:16.155157 140385973630784 torch/distributed/elastic/multiprocessing/api.py:858] Sending process 937054 closing signal SIGTERM
W1226 02:11:16.156132 140385973630784 torch/distributed/elastic/multiprocessing/api.py:858] Sending process 937055 closing signal SIGTERM
E1226 02:11:20.094970 140385973630784 torch/distributed/elastic/multiprocessing/api.py:833] failed (exitcode: 1) local_rank: 0 (pid: 937052) of binary: /usr/bin/python
Traceback (most recent call last):
  File "/usr/local/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/usr/local/lib/python3.8/dist-packages/accelerate/commands/accelerate_cli.py", line 48, in main
    args.func(args)
  File "/usr/local/lib/python3.8/dist-packages/accelerate/commands/launch.py", line 1161, in launch_command
    multi_gpu_launcher(args)
  File "/usr/local/lib/python3.8/dist-packages/accelerate/commands/launch.py", line 799, in multi_gpu_launcher
    distrib_run.run(args)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/run.py", line 892, in run
    elastic_launch(
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/launcher/api.py", line 133, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/launcher/api.py", line 264, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
train-original-script.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-12-26_02:11:16
  host      : x4-a100.internal.cloudapp.net
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 937052)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
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