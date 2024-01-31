#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import shutil
import platform
import sys
import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
import numpy as np
from PIL import Image

import numpy as np
import pandas as pd
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

import wandb 

from torchvision.transforms.functional import to_pil_image


import os
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import wandb
wandb.login(key="77a4e469d3a150b5ce7bd53d02214c79bea9f855")


if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__)

import torch
from torchvision import transforms
from diffusers import DiffusionPipeline

class CustomStableDiffusionPipeline(DiffusionPipeline):
    def __init__(self, text_encoder, vae, unet, tokenizer, noise_scheduler, size, *args, **kwargs):
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.size = size  # Consider reducing this if high resolution is not crucial

    def to(self, device):
        # Move components to device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        return self

    def preprocess(self, image, input_ids):
        image_tensor = self.image_to_tensor(image)
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        return image_tensor, input_ids

    def image_to_tensor(self, image):
        if torch.is_tensor(image):
            return F.interpolate(image, size=(self.size, self.size))
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            # Other transformations
        ])
        return transform(image)

    def __call__(self, image, text_prompt, num_inference_steps=None, generator=None, guidance_scale=None):
        image_tensor, input_ids = self.preprocess(image, text_prompt)
        post_disaster_image = self.generate_post_disaster_image(image_tensor, input_ids)

        # Clear unused memory after processing each image
        torch.cuda.empty_cache()

        return post_disaster_image

    def generate_post_disaster_image(self, pre_image_tensor, input_ids):
        encoder_hidden_states = self.text_encoder(input_ids)[0]
        encoded_output = self.vae.encode(pre_image_tensor)
        encoded_latents = encoded_output.latent_dist.sample()

        bsz = encoded_latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=encoded_latents.device)
        timesteps = timesteps.long()

        post_disaster_image = self.unet(encoded_latents, timesteps, encoder_hidden_states).sample
        return post_disaster_image

# Example usage:
# Assuming text_encoder, vae, unet, tokenizer, and noise_scheduler are already defined and loaded
#pipeline = CustomStableDiffusionPipeline(text_encoder, vae, unet, tokenizer, noise_scheduler)

# Example input data
#image = Image.open("path_to_pre_disaster_image.jpg")
#text_prompt = "Description of the desired post-disaster scenario"

# Generate post-disaster image
#post_disaster_image = pipeline(image, text_prompt)



def save_model_card(repo_id: str, base_model: str, images=None, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- pre-post-disaster
- diffusers
- disaster-response
inference: true
---
    """
    model_card = f"""
# Pre-Post Disaster Image Adaptation - {repo_id}
This model has been fine-tuned for pre- and post-disaster image analysis using the {base_model} as the base model. It is designed to understand and generate representations of disaster impact. Below are some example images generated by the model. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

from tqdm import tqdm  # Make sure to import tqdm

def log_validation(text_encoder, tokenizer, unet, vae, noise_scheduler, args, accelerator, weight_dtype, epoch, validation_dataset):
    logger.info(
        f"Running validation... \n Generating images for {len(validation_dataset)} pre-disaster scenarios."
    )
    #logging.debug("Starting validation...")

    # Create pipeline
    pipeline = CustomStableDiffusionPipeline(
        text_encoder=text_encoder, 
        vae=vae, 
        unet=unet, 
        tokenizer=tokenizer, 
        noise_scheduler=noise_scheduler,
        size=512
    )
    pipeline.set_progress_bar_config(disable=True)

    # Run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    validation_images = []

    # Initialize progress bar
    progress_bar = tqdm(total=len(validation_dataset), desc="Validating", leave=False)

    for i in range(len(validation_dataset)):
        pre_image_data = validation_dataset[i]
        pre_image_text = pre_image_data['input_ids']
        pre_image = pre_image_data['pre_pixel_values'].unsqueeze(0).to(accelerator.device)

        # Run model prediction
        with torch.autocast("cuda"):
            post_image_generated = pipeline(pre_image, pre_image_text)

        # Convert tensors to PIL Images and save
        pre_image_pil = to_pil_image(pre_image[0].cpu())
        if post_image_generated.ndim == 4:
            image_tensor = post_image_generated.squeeze(0)  # Remove the batch dimension
        post_image_pil = to_pil_image(image_tensor.cpu())

        save_directory = "./finaldata/testimgs/"
        pre_image_pil.save(os.path.join(save_directory, f"pre_disaster_{i}.png"))
        post_image_pil.save(os.path.join(save_directory, f"post_disaster_generated_{i}.png"))

        # Clear memory
        torch.cuda.empty_cache()

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()  # Close the progress bar after processing

        
    # Log images
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i, (pre_img, post_img_gen) in enumerate(validation_images):
                tracker.writer.add_image(f"validation_pre_{i}", pre_img, epoch, dataformats="CHW")
                tracker.writer.add_image(f"validation_post_generated_{i}", post_img_gen, epoch, dataformats="CHW")
        if tracker.name == "wandb":
            for i, (pre_img, post_img_gen) in enumerate(validation_images):
                tracker.log({
                    f"validation_pre_{i}": wandb.Image(pre_img, caption=f"Pre-Disaster: {i}"),
                    f"validation_post_generated_{i}": wandb.Image(post_img_gen, caption=f"Post-Disaster Generated: {i}")
                })

    del pipeline
    torch.cuda.empty_cache()
    return validation_images

def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for pre- and post-disaster image analysis.")
    
    parser.add_argument(
        "--pre_disaster_csv_file",
        type=str,
        default=None,
        required=True,
        help="File containing pre-disaster prompts.")
        
    parser.add_argument(
        "--post_disaster_csv_file",
        type=str,
        default=None,
        required=True,
        help="File containing post-disaster prompts.")
    
    
    parser.add_argument(
        "--pre_disaster_data_dir",
        type=str,
        default=None,
        required=True,
        help="Directory containing pre-disaster images.")
    
    parser.add_argument(
        "--post_disaster_data_dir",
        type=str,
        default=None,
        required=True,
        help="Directory containing post-disaster images.")
    
    parser.add_argument(
        "--validation_prompt_pre",
        type=str,
        default=None,
        help="A prompt used during validation for pre-disaster scenarios.",
    )
    parser.add_argument(
        "--validation_prompt_post",
        type=str,
        default=None,
        help="A prompt used during validation for post-disaster scenarios.",
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.pre_disaster_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

disaster_templates = [
    "a satellite image of a region before the disaster {}",
    "an aerial view of a landscape prior to the calamity {}",
    "a pre-disaster scene showing the area {}",
    "a satellite view of the region after the disaster {}",
    "post-disaster imagery of the affected zone {}",
    "a landscape following the disaster event {}"
]

def load_image_as_numpy_array(image_path):
    with Image.open(image_path) as img:
        return np.array(img)


class DisasterImageDataset(Dataset):
    def __init__(
        self,
        pre_data_dir,
        post_data_dir,
        pre_csv_file,
        post_csv_file,
        tokenizer,
        size=512,
        interpolation="bicubic",
        flip_p=0.5,
        center_crop=False,
    ):
        self.pre_data_dir = pre_data_dir
        self.post_data_dir = post_data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        # Load descriptions from CSV files
        self.pre_descriptions = pd.read_csv(pre_csv_file).set_index('File Name')['Description'].to_dict()
        self.post_descriptions = pd.read_csv(post_csv_file).set_index('File Name')['Description'].to_dict()

        # List of pre-disaster image paths
        self.pre_image_paths = [os.path.join(pre_data_dir, file) for file in os.listdir(pre_data_dir)]
        self.post_image_paths = [os.path.join(post_data_dir, file) for file in os.listdir(post_data_dir)]

        self._length = len(self.pre_image_paths)

        self.interpolation = transforms.InterpolationMode.BICUBIC if interpolation == "bicubic" else transforms.InterpolationMode.BILINEAR
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        pre_image_path = self.pre_image_paths[idx]
        pre_image_name = os.path.basename(pre_image_path)
        post_image_name = pre_image_name.replace('pre', 'post')
        post_image_path = os.path.join(self.post_data_dir, post_image_name)

        pre_image = load_image_as_numpy_array(pre_image_path)
        post_image = load_image_as_numpy_array(post_image_path)
        # Process images
        pre_image, post_image = self.process_image(pre_image), self.process_image(post_image)

        # Get descriptions
        pre_text = self.pre_descriptions.get(pre_image_name, "")
        post_text = self.post_descriptions.get(post_image_name, "")
        text = f"Pre Disaster: {pre_text}; Post Disaster: {post_text}"
        
        pre_tensor = pre_image
        post_tensor = post_image

        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "input_ids": input_ids,
            "pre_pixel_values": pre_tensor,
            "post_pixel_values": post_tensor,
        }

    def process_image(self, image):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Image processing (resize, crop, flip, etc.)
        transform = transforms.Compose([
            transforms.CenterCrop(self.size) if self.center_crop else transforms.Resize((self.size, self.size), interpolation=self.interpolation),
            self.flip_transform,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return transform(image)
    
# Example usage:
# dataset = DisasterImageDataset(pre_data_root='path_to_pre_images', 
#                                post_data_root='path_to_post_images',
#                                pre_csv_file='path_to_pre_csv',
#                                post_csv_file='path_to_post_csv',
#                                tokenizer=your_tokenizer)

def save_custom_pipeline(pipeline, save_directory):
    # Check if the models are wrapped with DDP and unwrap them
    text_encoder = pipeline.text_encoder.module if hasattr(pipeline.text_encoder, 'module') else pipeline.text_encoder
    vae = pipeline.vae.module if hasattr(pipeline.vae, 'module') else pipeline.vae
    unet = pipeline.unet.module if hasattr(pipeline.unet, 'module') else pipeline.unet

    # Save each component of the pipeline
    text_encoder.save_pretrained(save_directory)
    vae.save_pretrained(save_directory)
    unet.save_pretrained(save_directory)
    pipeline.tokenizer.save_pretrained(save_directory)
    # Save any other components or configurations as needed


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # Add the new lines here for Weights & Biases initialization
    import wandb
    wandb.init(project="textual_inversion_img2img", dir=args.output_dir)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    validation_dataset = DisasterImageDataset(
        pre_data_dir="./finaldata/pre_val_test/",
        post_data_dir="./finaldata/post_val_test/",  # If available
        pre_csv_file="./finaldata/val_pre_hurricane_prompts.csv",
        post_csv_file="./finaldata/val_post_hurricane_prompts.csv",  # If available
        tokenizer=tokenizer
    )

    # Dataset and DataLoaders creation:
    train_dataset = DisasterImageDataset(
        pre_data_dir=args.pre_disaster_data_dir,
        post_data_dir=args.post_disaster_data_dir,
        pre_csv_file="./finaldata/pre_hurricane_prompts.csv",
        post_csv_file="./finaldata/post_hurricane_prompts.csv",
        tokenizer=tokenizer,
        #size=args.resolution,
        #placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
        #repeats=args.repeats,
        #learnable_property=args.learnable_property,
        #center_crop=args.center_crop,
        #set="train",
    )
    print("Batch size:", args.train_batch_size)
    print("DataLoader Num Workers:", args.dataloader_num_workers)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size // accelerator.num_processes,  # Adjust batch size
        shuffle=True, 
        num_workers=args.dataloader_num_workers
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    from importlib.metadata import distributions

    def get_installed_packages():
        return {dist.metadata['Name']: dist.version for dist in distributions()}
    
    # Add the new lines here
    import json
    
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    training_params = {
        "total_batch_size": args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
        "batch_size": args.train_batch_size // accelerator.num_processes,
        "learning_rate": args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes,
        "num_train_epochs": math.ceil(args.max_train_steps / num_update_steps_per_epoch),
        # ... other parameters ...
    }

    with open(os.path.join(args.output_dir, 'training_parameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    environment_info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "installed_packages": get_installed_packages(),
    }
    
    with open(os.path.join(args.output_dir, 'environment_info.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Encode pre and post disaster images to their respective latents
                pre_latents = vae.encode(batch["pre_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                post_latents = vae.encode(batch["post_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()

                # Combine the pre and post latents - example here uses addition
                combined_latents = pre_latents + post_latents

                # Sample noise to add to the combined latents
                noise = torch.randn_like(combined_latents)
                bsz = combined_latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=combined_latents.device)
                timesteps = timesteps.long()

                # Add noise to the combined latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(combined_latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(combined_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]
                    

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"learned_embeds-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds-steps-{global_step}.safetensors"
                    )
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, "learned_embeddings.bin")
                        save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path)

                        # Saving model state
                        state_save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(state_save_dir)
                        logger.info(f"Saved state to {state_save_dir}")
                        
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            text_encoder, tokenizer, unet, vae, noise_scheduler, args, accelerator, weight_dtype, epoch, validation_dataset
                        )
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break

        # Add these lines to save the model components
    # Unwrap the models from DDP wrapper
    original_text_encoder = accelerator.unwrap_model(text_encoder)
    original_vae = accelerator.unwrap_model(vae)
    original_unet = accelerator.unwrap_model(unet)

    # Save the unwrapped models
    original_text_encoder.save_pretrained(args.output_dir)
    original_vae.save_pretrained(args.output_dir)
    original_unet.save_pretrained(args.output_dir)

    # If saving tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline

        if save_full_model:
            custom_pipeline = CustomStableDiffusionPipeline(
                text_encoder, vae, unet, tokenizer, noise_scheduler, size = 512
            )
            save_custom_pipeline(custom_pipeline, args.output_dir)

        # Save the newly trained embeddings
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("./error_file.txt", "w") as file:
            file.write(str(e))
        raise