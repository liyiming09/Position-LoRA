import argparse
import json
import logging
import math, cv2
import os
import random
from pathlib import Path
from typing import Optional
from functools import partial
import diffusers
import numpy as np
import PIL
import torch, itertools
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from templates.stop_words import stop_words
from attn_guidance import (AttnStorage, CustomAttnProcessor, prepare_attention, get_attn_dict, get_attns)
from regular_dataset import ReVersionRegularDataset, ReVersionRegularDatasetnoRandom, ReVersionandRegularDataset, ReVersionandRegularDatasetv2, ReVersionandRegularDatasetv3
from dp_dataset import DPDatasetnoRegular, DPDatasetnoRegularv2
from lora import (
    save_lora_weight,
    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    get_target_module,
    save_lora_layername,
    monkeypatch_or_replace_lora,
    monkeypatch_remove_lora,
    set_lora_requires_grad,
)
from collections import OrderedDict
from utils.clipseg import CLIPDensePredT
from NullextInversion import NullInversion
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if version.parse(version.parse(
        PIL.__version__).base_version) >= version.parse("9.1.0"):
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
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif')


def is_image_file(filename):
    # return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    return filename.endswith(IMG_EXTENSIONS)


def save_progress(text_encoder, placeholder_token_id, accelerator, args,
                  save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(
        text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {
        args.placeholder_token: learned_embeds.detach().cpu()
    }
    torch.save(learned_embeds_dict, save_path)
    # torch.save(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[-1:].detach().clone(),'{}_r_attn_loraemb{}_{}.pth'.format(args.train_data_dir.split('/')[-1], str(args.src_lr),args.initializer_token))
    
        

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default='ddim',
        help="mode to get the initial latent noise, such as random, DDIM or else.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='/internfs/xxxxx/huggingface_models/stable-diffusion-v1-5',
        # required=True,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=
        "Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='./reversion_benchmark_v1/shake_hands',
        # required=True,
        help=
        "The folder that contains the exemplar images (and coarse descriptions) of the specific relation."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<R>",
        # required=True,
        help="A token to use as a placeholder for the relation.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default='and',
        # required=True,
        help="A token to use as initializer word.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./experiments/test',
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=
        ("The resolution for input images, all the images in the train/validation dataset will be resized to this"
         " resolution"),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-05,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--noise_learning_rate",
        type=float,
        default=5e-05,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=1e-05,
        help=
        "Initial lora learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=
        ("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
         ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help=
        "The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=
        ('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
         ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
         ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=
        "A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help=
        "Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=
        ("Run validation every X epochs. Validation consists of running the prompt"
         " `args.validation_prompt` multiple times: `args.num_validation_images`"
         " and logging the images."),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.")

    parser.add_argument(
        "--importance_sampling",
        action='store_true',
        default=False,
        help="Relation-Focal Importance Sampling",
    )
    parser.add_argument(
        "--denoise_loss_weight",
        type=float,
        default=1.0,
        help="Weight of L_denoise",
    )
    parser.add_argument(
        "--noise_loss_weight",
        type=float,
        default=0.1,
        help="Weight of L_denoise",
    )
    parser.add_argument(
        "--regular_weight",
        type=float,
        default=0.5,
        help="Weight of regularization_denoise",
    )
    parser.add_argument(
        "--steer_loss_weight",
        type=float,
        default=0.01,
        help="Weight of L_steer (for Relation-Steering Contrastive Learning)",
    )
    parser.add_argument(
        "--attn_loss_weight",
        type=float,
        default=0.0,
        help="Weight of L_attn(for INDEPENDENCE Attention Contrastive Learning)",
    )
    parser.add_argument(
        "--num_positives",
        type=int,
        default=4,
        help="Number of positive words used for L_steer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default="0.07",
        help="Temperature parameter for L_steer",
    )
    parser.add_argument(
        "--scaled_cosine_alpha",
        type=float,
        default=0.5,
        help="The skewness (alpha) of the Importance Sampling Function",
    )
    parser.add_argument(
        "--sds",
        action="store_true",
        help=
        ("Whether or not to use a SDS method in DreamFusion to optimize the parameters in Lora and word-embedding"
         ),
    )
    parser.add_argument(
        "--dds",
        action="store_true",
        help=
        ("Whether or not to use a DDS method in DreamFusion to optimize the parameters in Lora and word-embedding"
         ),
    )
    parser.add_argument(
        "--sds_loss_weight",
        type=float,
        default=0.05,
        help="Weight of sds loss",
    )
    parser.add_argument(
        "--regular_thres",
        type=float,
        default=0.1,
        help="threshold for regular datas in dataset",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        help="to do a dp image training or a natural relation training",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
    )
    parser.add_argument(
        "--mix_noise_weight",
        type=float,
        default=0.8,
        help="Weight of mix noise weight for the input noise",
    )
    
    parser.add_argument(
        "--different",
        action="store_true",
        help='to make the input noise and target noise different, to force lora learn an additional structural noise'
    )
    parser.add_argument(
        "--crop_p",
        type=float,
        default=0.3,
        help="Whether to crop images before resizing to resolution.")
    parser.add_argument(
        "--flip_p",
        type=float,
        default=0.3,
        help="Whether to flip images before resizing to resolution.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def calculate_attn_loss(attn_storage, input_ids, special_ids, placeholder_token_id = 49408):
    bs = input_ids.shape[0]
    attns = get_attns(attn_storage)
    h = attns[0].shape[0]//bs
    R_mask = torch.isin(input_ids,torch.tensor([placeholder_token_id]).cuda())
    other_mask = torch.isin(input_ids,torch.tensor(special_ids + [placeholder_token_id]).cuda())

    loss = 0 # can only handle the tensor with bs = 1, if not, we need to fix it.
    for i in range(len(attns)):
        tmp_attn = attns[i].reshape(bs, h, *attns[i].shape[1:]).permute(0,3,2,1)
        for b in range(bs):
            R_attn = tmp_attn[b:b+1][R_mask[b:b+1]].reshape(1, -1)
            other_attns = tmp_attn[b:b+1][~other_mask[b:b+1]]
            other_attns = other_attns.reshape(other_attns.shape[0],-1)
            other_attn = other_attns.sum(0,keepdims=True) / other_attns.shape[0]
            norm_R_attn = F.normalize(R_attn, p=2, dim=1)
            norm_other_attn = F.normalize(other_attn, p=2, dim=1)

            loss += (norm_R_attn*norm_other_attn).sum()
    return loss

def calculate_reg_attn_loss(attn_storage, input_ids, special_ids, placeholder_token_id = 49408):
    bs = input_ids.shape[0]
    attns = get_attns(attn_storage)
    h = attns[0].shape[0]//bs
    R_mask = torch.isin(input_ids,torch.tensor([placeholder_token_id]).cuda())
    # other_mask = torch.isin(input_ids,torch.tensor(special_ids + [placeholder_token_id]).cuda())

    loss = 0 # can only handle the tensor with bs = 1, if not, we need to fix it.
    for i in range(len(attns)):
        tmp_attn = attns[i].reshape(bs, h, *attns[i].shape[1:]).permute(0,3,2,1)
        for b in range(bs):
            R_attn = tmp_attn[b:b+1][R_mask[b:b+1]].reshape(1, -1)
            R_norm = R_attn.max()

            loss += ((R_attn/R_norm)**2).mean()
    return loss

def calculate_LAC_loss(attn_storage, input_ids, special_ids, regular_mask, placeholder_token_id = 49408):
    # LAC loss in LoCo:
    bs = input_ids.shape[0]
    attns = get_attns(attn_storage)
    h = attns[0].shape[0]//bs
    R_mask = torch.isin(input_ids,torch.tensor([placeholder_token_id]).cuda())
    # other_mask = torch.isin(input_ids,torch.tensor(special_ids + [placeholder_token_id]).cuda())

    loss = 0 # can only handle the tensor with bs = 1, if not, we need to fix it.
    for i in range(len(attns)):
        
        tmp_attn = attns[i].reshape(bs, h, *attns[i].shape[1:]).permute(0,3,2,1)
        map_size = int(math.sqrt(tmp_attn.shape[2]))
        mode = 'area' if regular_mask.shape[3] >= map_size else 'bicubic'
        resize_mask = torch.nn.functional.interpolate(regular_mask, size = [map_size, map_size], mode = mode) # shape: 1,1,64,64
                    
        for b in range(bs):
            R_attn = tmp_attn[b:b+1][R_mask[b:b+1]].reshape(1, map_size,map_size,h).permute(0,3,1,2)
            # img = R_attn.sum(1) / R_attn.shape[1]
            # img = img[0] / img.max()
            # img = 255 * img
            # img = img.cpu().numpy().astype(np.uint8)
            # cv2.imwrite('Rattn.png',img)
            R_norm = R_attn.max()
            mask_R_attn = R_attn * (1-resize_mask)
            tmp_loss = (1 - (mask_R_attn/R_norm).sum()/(R_attn/R_norm).sum()) ** 2

            loss += tmp_loss
    return loss

def calculate_steer_loss(token_embedding,
                         input_ids,
                         placeholder_token_id,
                         stop_ids,
                         special_ids,
                         positive_ids,
                         temperature=0.07):
    """L_steer"""
    # compute input embeddings
    inputs_embeds = token_embedding(input_ids)  # (bs, 77, 768) #输入text的token id
    positive_embeds = token_embedding(positive_ids)

    with torch.no_grad(
    ):  # no gradients from positive and negative embeds, only from <R>
        # compute entity embeds
        stop_mask = torch.isin(
            input_ids,
            torch.tensor(stop_ids + special_ids +
                         [placeholder_token_id]).cuda())  # (bs, 77)
        negative_embds = inputs_embeds[~stop_mask]  # (num_stop_tokens, 768)

        # remove bos and eos in positive embeddings
        stop_mask = torch.isin(positive_ids,
                               torch.tensor(special_ids).cuda())  # (bs, 77)
        positive_embeds = positive_embeds[
            ~stop_mask]  # (num_positive_tokens, 768), where num_positive_tokens = num_positives * bs

        # stack positives and negatives as a pn_block
        pn_embeds = torch.cat([positive_embeds, negative_embds], dim=0)
        pn_embeds_normalized = F.normalize(
            pn_embeds, p=2,
            dim=1)  # (num_positive_tokens+num_negative_tokens, 768)

    # compute relation embeds <R>
    relation_mask = (input_ids[0] == placeholder_token_id)  # (77)
    relation_embeds = inputs_embeds[0][relation_mask]  # (1, 768)
    relation_embeds_normalized = F.normalize(relation_embeds, p=2, dim=1)

    # compute Multi-Instance InfoNCE loss
    logits = torch.einsum('nc,mc->nm',
                          [relation_embeds_normalized, pn_embeds_normalized
                           ])  # (1, num_positive_tokens+num_negative_tokens)

    logits /= temperature
    nominator = torch.logsumexp(logits[:, :positive_embeds.shape[0]], dim=1)
    denominator = torch.logsumexp(logits, dim=1)

    return torch.mean(denominator - nominator)


def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))
  

def main():

    args = parse_args()

    from templates.relation_words import relation_words

    print(f'args.learning_rate={args.learning_rate}')
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=
        logging_dir,  # logging_dir=logging_dir, # depends on accelerator vesion
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

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
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(
                args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"),
                      "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision)
    if args.sds:
        guidemodel = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)

            # non-strict, because we only stored decoder weights (not CLIP weights)
        guidemodel.load_state_dict(torch.load('./utils/weights/rd64-uni.pth', map_location=accelerator.device), strict=False)
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer.")

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(
        args.placeholder_token)
    


    # stop words id
    expanded_stop_words = stop_words + relation_words  # add relation words to stop_words
    stop_ids = tokenizer(
        " ".join(expanded_stop_words),
        truncation=False,
        return_tensors="pt",
    ).input_ids[0].detach().tolist()

    # stop_ids = stop_ids + [tokenizer.bos_token_id, tokenizer.eos_token_id] # add special token ids to stop ids
    special_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id]

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(True)
    if args.sds:
        guidemodel.requires_grad_(False)
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
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    args.src_lr = args.learning_rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes)

    # define lora parameters to train
    text_encoder_lora_params = None
    unet_lora_params = None
    use_lora_extended = False
    lora_unet_rank = 32
    lora_txt_rank = 32
    injectable_lora = get_target_module("injection", use_lora_extended)
    target_module = get_target_module("module", use_lora_extended)

    unet_lora_params, _ = injectable_lora(
            unet,
            r=lora_unet_rank,
            loras=None,
            target_replace_module=target_module,
        )
    save_lora_layername(unet)

    if args.dp:
        train_dataset = DPDatasetnoRegularv2(
            data_root=args.train_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            placeholder_token=args.placeholder_token,
            repeats=args.repeats,
            crop_p=args.crop_p,
            flip_p=args.flip_p,
            set="train",
            relation_words=relation_words,
            num_positives=args.num_positives)
    else:
        # Dataset and DataLoaders creation:
        train_dataset = ReVersionandRegularDatasetv2(
            data_root=args.train_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            placeholder_token=args.placeholder_token,
            repeats=args.repeats,
            center_crop=args.center_crop,
            set="train",
            relation_words=relation_words,
            num_positives=args.num_positives,
            regular_thres = args.regular_thres
            )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers)

    # define initial latent noise to train
    # 初始化权重矩阵，这里使用正态分布初始化
    dataset_length = train_dataset.num_images

    local_f = open(os.path.join(args.train_data_dir, 'text.json'))
    templates = json.load(local_f, object_pairs_hook=OrderedDict)
    noise_dict = {}
    for index, (key, value) in enumerate(templates.items()):
        noise_dict[key] = index
    
    if args.noise_mode == 'random':
        noise_embedding_weight = torch.nn.Parameter(torch.randn(dataset_length, 4, 64, 64, device=accelerator.device, dtype=torch.float32), requires_grad=True)
        # noise_embedding_weight = noise_embedding_weight.to(accelerator.device, dtype=torch.float32)
        orig_noise_embeds = noise_embedding_weight.detach().clone()
    else:
        noise_embedding_dicts = {}
        if os.path.exists(os.path.join(args.train_data_dir, 'bbox.json')):
            bbox_f = open(os.path.join(args.train_data_dir, 'bbox.json'))
            bboxes = json.load(bbox_f, object_pairs_hook=OrderedDict)
        else:
            bboxes = {}
            for index, (key, value) in enumerate(templates.items()):
                bboxes[key] = [[0, 0, 512, 512]]
        if args.train_data_dir[-1] == '/': args.train_data_dir = args.train_data_dir[:-1] 
        noise_embeds_root = args.train_data_dir + '_noise'
        for index, (key, value) in enumerate(templates.items()):
            cur_noise_embed = torch.load(os.path.join(noise_embeds_root, key.replace('png', 'pth').replace('jpg', 'pth'))  ,map_location = accelerator.device)
            noise_embedding_dicts[key] = cur_noise_embed
        orig_noise_embeds = None

    
    
    
    params_to_optimize = [
                {
                    "params": itertools.chain(*unet_lora_params),
                    "lr": args.lora_learning_rate,
                    "grad_clip_norm": 0.5
                },
                # {
                #     "params": noise_embedding_weight,
                #     "lr": args.noise_learning_rate,
                # },
                {
                    "params": text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
                    "lr": args.learning_rate,
                },
            ]

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        # lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )



    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler)

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    if args.sds:
        guidemodel.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
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
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    # Relation-Focal Importance Sampling
    if args.importance_sampling:
        print("Using Relation-Focal Importance Sampling")
        list_of_candidates = [
            x for x in range(noise_scheduler.config.num_train_timesteps)
        ]
        prob_dist = [
            importance_sampling_fn(x,
                                   noise_scheduler.config.num_train_timesteps,
                                   args.scaled_cosine_alpha)
            for x in list_of_candidates
        ]
        prob_sum = 0
        # normalize the prob_list so that sum of prob is 1
        for i in prob_dist:
            prob_sum += i
        prob_dist = [x / prob_sum for x in prob_dist]

    # set up the custom attn processor and use to replace standard model processors
    if args.attn_loss_weight > 0: 
        storage = AttnStorage()
        processor = partial(CustomAttnProcessor, storage)
        attn_dict = get_attn_dict(processor, unet)
        unet.set_attn_processor(attn_dict)



    method =  'noise_astar_diffe_0317_'+args.train_data_dir.split('/')[-1] if args.different else 'noise_astar'
    out_root = '{}/n{}_a{}_m{}'.format(method, args.noise_loss_weight,args.attn_loss_weight, args.mix_noise_weight)  + ('_detach' if  args.detach else '')
    sub_root = '{}_t{}_lora{}/'.format(args.train_data_dir.split('/')[-1], str(args.learning_rate), str(args.lora_learning_rate))
    if not os.path.exists(os.path.join(out_root, sub_root)): os.makedirs(os.path.join(out_root, sub_root))
    # torch.save(orig_noise_embeds.detach().clone(),'./{}/{}/{}_noisemb_ori.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))

    # noise_embedding_weight = noise_embedding_weight.requires_grad_()
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                if batch["is_regular"][0]:
                    latents = vae.encode(batch["regular_img"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                else:
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # timestep (t) sampling
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps, (bsz, ),
                    device=latents.device)
                # Relation-Focal Importance Sampling
                if args.importance_sampling:
                    timesteps = np.random.choice(
                        list_of_candidates,
                        size=bsz,
                        replace=True,
                        p=prob_dist)
                    timesteps = torch.tensor(timesteps).cuda()
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning

                if batch["is_regular"][0]:
                    encoder_hidden_states = text_encoder(batch["regular_ids"])[0].to(dtype=weight_dtype)
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                # R_index = batch["input_ids"].index(placeholder_token_id)
                # Predict the noise residual
                if args.attn_loss_weight > 0: prepare_attention(unet, storage, set_store=False)

                # Get the target for loss depending on the prediction type
                

                if args.attn_loss_weight > 0: prepare_attention(unet, storage, pred_type='orig', set_store=True)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample


                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = 0.0

                # L_denoise
                '''1. DDPM loss for a common trainging'''
                if batch["is_regular"][0]:
                    # resize_mask = torch.nn.functional.interpolate(batch["regular_mask"], size = [64, 64], mode = 'area')
                    # denoise_loss = F.mse_loss( (model_pred * resize_mask).float(), (target * resize_mask).float(), reduction="mean")
                    denoise_loss = F.mse_loss( model_pred.float(), target.float(), reduction="mean")
                    weighted_denoise_loss = args.regular_weight * args.denoise_loss_weight * denoise_loss
                else:
                    if args.different:
                        if args.dp:
                            tar_noise = batch["latent_emb"]
                            target = args.mix_noise_weight * tar_noise \
                                     + (1 - args.mix_noise_weight) * noise
                    
                        else:
                            for b in range(bsz):
                                name_b = batch["name"][b]
                                h, w = batch["original_sizes"][b][0], batch["original_sizes"][b][1]
                                # print(h,w)
                                for bbox in bboxes[name_b]:
                                    min_height, min_width, max_height, max_width = bbox
                                    min_height = torch.round(args.resolution/h*min_height)
                                    max_height = torch.round(args.resolution/h*max_height)
                                    min_width = torch.round(args.resolution/w*min_width)
                                    max_width = torch.round(args.resolution/w*max_width)
                                    tar_noise = noise_embedding_dicts[name_b].detach()
                                    target[b,:,min_height//8:max_height//8, min_width//8:max_width//8] = args.mix_noise_weight * tar_noise[0,:,min_height//8:max_height//8, min_width//8:max_width//8] \
                                                                                                        + (1 - args.mix_noise_weight) * noise[b,:,min_height//8:max_height//8, min_width//8:max_width//8]
                    
                    denoise_loss = F.mse_loss( model_pred.float(), target.float(), reduction="mean")
                    weighted_denoise_loss = args.denoise_loss_weight * denoise_loss
                loss += weighted_denoise_loss

                '''2. use a DDIM way to reconstruct \hat_z_T, to optimize the latent noise embeds'''
                # if not batch["is_regular"][0]:
                #     a_t = noise_scheduler.alphas_cumprod

                #     alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
                #     to_torch = lambda x: x.to(alphas_cumprod.dtype).to(accelerator.device)
                #     # alphas_prev_cumprod = torch.tensor(np.append(1., alphas_cumprod[:-1])).to(alphas_cumprod.dtype).to(accelerator.device)
                #     sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
                #     a_t, sqrt_one_minus_at = [], []
                #     for b in range(bsz):
                #         a_t.append(torch.full((1, 1, 1, 1), alphas_cumprod[timesteps[b]], device=accelerator.device))
                #         # a_prev.append(torch.full((1, 1, 1, 1), alphas_prev_cumprod[timesteps[b]], device=device))
                #         sqrt_one_minus_at.append(torch.full((1, 1, 1, 1), sqrt_one_minus_alphas_cumprod[timesteps[b]],device=accelerator.device))
                #     a_t = torch.cat(a_t,dim = 0)
                #     sqrt_one_minus_at = torch.cat(sqrt_one_minus_at,dim = 0)
                    
                #     pred_z0 = (noisy_latents - sqrt_one_minus_at * model_pred) / a_t.sqrt()
                #     pred_zT = torch.sqrt(alphas_cumprod[-1]) * pred_z0 + torch.sqrt(1-alphas_cumprod[-1]) * model_pred

                #     indice = []
                #     resize_mask = torch.nn.functional.interpolate(batch["regular_mask"], size = [64, 64], mode = 'nearest') # shape: 1,1,64,64
                #     for b in range(bsz):
                #         indice.append(noise_dict[batch["name"][b]])
                #     noise_loss = F.mse_loss( (noise_embedding_weight[indice]*(1-resize_mask)).float(),(pred_zT*(1-resize_mask)).float().detach() if args.detach else pred_zT.float(),  reduction="mean")
                #     weighted_noise_loss = args.noise_loss_weight * noise_loss

                #     loss += weighted_noise_loss
                    

                '''3. Steering loss for a semantic contrastive training'''
                token_embedding = accelerator.unwrap_model(
                    text_encoder).get_input_embeddings()  # with grad

                # L_steer
                if (not batch["is_regular"][0]) and args.steer_loss_weight > 0:
                    assert args.num_positives > 0
                    steer_loss = calculate_steer_loss(
                        token_embedding,
                        batch["input_ids"],
                        placeholder_token_id,
                        stop_ids,
                        special_ids,
                        batch["positive_ids"],
                        temperature=args.temperature)
                    weighted_steer_loss = args.steer_loss_weight * steer_loss
                    loss += weighted_steer_loss

                '''4.  Attention Guidance Loss: to optimize the degree of independence of <R>'s attn '''
                if args.attn_loss_weight >0:
                    if (not batch["is_regular"][0]):
                        attn_loss = calculate_LAC_loss(storage, batch["input_ids"], special_ids,batch["regular_mask"], placeholder_token_id)
                        weighted_attn_loss = args.attn_loss_weight * attn_loss
                        loss += weighted_attn_loss
                    # else:
                    #     attn_loss = calculate_reg_attn_loss(storage, batch["input_ids"], special_ids, placeholder_token_id)
                    #     weighted_attn_loss = args.attn_loss_weight * attn_loss
                    #     loss += weighted_attn_loss

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.arange(
                    len(tokenizer)) != placeholder_token_id
                with torch.no_grad():
                    accelerator.unwrap_model(
                        text_encoder).get_input_embeddings(
                        ).weight[index_no_updates] = orig_embeds_params[
                            index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(
                        args.output_dir,
                        f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id,
                                  accelerator, args, save_path)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir,
                                                 f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                # "lr": lr_scheduler.get_last_lr()[0],
                "loss": loss.detach().item(),
                # "denoise_loss": denoise_loss.detach().item(),
                "weighted_denoise_loss": weighted_denoise_loss.detach().item(),
            }
            if args.attn_loss_weight > 0:
                logs["weighted_attn_loss"] = weighted_attn_loss.detach().item()
            if args.steer_loss_weight > 0:
                logs["steer_loss"] = steer_loss.detach().item()
                logs["weighted_steer_loss"] = weighted_steer_loss.detach().item()
            
            # if (args.sds or args.dds) and args.sds_loss_weight > 0:
            #     logs["weighted_sds_loss"] = weighted_sds_loss.detach().item()



            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # validation
        if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
            logger.info(
                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                f" {args.validation_prompt}.")
            # create pipeline (note: unet and vae are loaded again in float32)
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                unet=unet,
                vae=vae,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config)
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = (None if args.seed is None else torch.Generator(
                device=accelerator.device).manual_seed(args.seed))
            images = []
            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(
                        args.validation_prompt,
                        num_inference_steps=25,
                        generator=generator).images[0]
                images.append(image)

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "validation", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log({
                        "validation": [
                            wandb.Image(
                                image,
                                caption=f"{i}: {args.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    })

            del pipeline
            torch.cuda.empty_cache()

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and args.only_save_embeds:
            logger.warn(
                "Enabling full model saving because --push_to_hub=True was specified."
            )
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        # save_progress(text_encoder, placeholder_token_id, accelerator, args,
                    #   save_path)
        
        torch.save(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[-1:].detach().clone(),'./{}/{}/{}_loraemb.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))
        # torch.save(noise_embedding_weight.detach().clone(),'./{}/{}/{}_noisemb.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))
        
        save_lora_weight(unet,'./{}/{}/{}_lora.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]),save_safetensors = False)
        # save_lora_weight(unet,'./{}/{}/{}_lora.pt'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]),save_safetensors = True)
        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training",
                blocking=False,
                auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
