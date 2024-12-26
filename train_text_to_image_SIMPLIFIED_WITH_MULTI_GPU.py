from datasets import load_dataset
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch import autocast, GradScaler
from accelerate import Accelerator
import torch.nn.functional as F


mixed_precision = "fp16"
gradient_accumulation_steps = 1
max_grad_norm = 1.0

accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    mixed_precision=mixed_precision,
)
device = accelerator.device


max_train_steps = 10
learning_rate = 1e-5
train_batch_size = 1
resolution = 256
weight_dtype = torch.float16
dataset_name = "reach-vb/pokemon-blip-captions"
dataset = load_dataset(dataset_name)

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(
    model_id,
    subfolder='vae',
    in_channels=1,
    out_channels=1,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    latent_channels=2
)
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    in_channels=2,
    out_channels=2,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
)
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
scaler = GradScaler()
num_training_steps_for_scheduler = max_train_steps
lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_training_steps=num_training_steps_for_scheduler,
)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def preprocess_train(examples):
    images = [image.convert("L") for image in examples["image"]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples

train_dataset = dataset["train"].with_transform(preprocess_train)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=train_batch_size,
)

unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)
text_encoder.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples["text"]:
            captions.append(caption)
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


train_transforms = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


if __name__ == "__main__":
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        if step >= max_train_steps:
            break

        with accelerator.accumulate(unet):

            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
            target = noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps
    
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("Training is finished. Generating a sample image...")

    with torch.autocast(accelerator.device.type):
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=accelerator.unwrap_model(unet),
            safety_checker = None,    
        )
        generator = torch.Generator(device='cpu').manual_seed(1)
        image = pipeline("Photo of a worm", num_inference_steps=20, generator=generator).images[0]
    image.save("sample_image.png")