from datasets import load_dataset
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch import autocast, GradScaler


max_train_steps = 10
learning_rate = 1e-5
train_batch_size = 1
resolution = 256
mixed_precision = "fp16"
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
unet.to("cuda")
text_encoder.to("cuda", dtype=torch.float16)
vae.to("cuda", dtype=torch.float16)

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

def preprocess_train(examples):
    images = [image.convert("L") for image in examples["image"]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples


train_dataset = dataset["train"].with_transform(preprocess_train)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=train_batch_size,
)

num_training_steps_for_scheduler = max_train_steps

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_training_steps=num_training_steps_for_scheduler,
)


if __name__ == "__main__":
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        if step >= max_train_steps:
            break
    
        pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
        input_ids = batch["input_ids"].to("cuda")
    
        with autocast("cuda"):
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=latents.device).long()
    
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids)[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(model_pred, noise)
    
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        train_loss += loss.item()
    
    unet.to("cuda", dtype=torch.float16)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        safety_checker = None,
        torch_dtype=torch.float16,
    
    ).to("cuda")
    generator = torch.Generator(device='cpu').manual_seed(1)
    image = pipeline("Photo of a worm", num_inference_steps=20, generator=generator).images[0]
    image.save("sample_image.png")