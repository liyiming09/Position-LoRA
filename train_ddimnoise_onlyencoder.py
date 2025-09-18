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
from regular_dataset import (ReVersionRegularDataset, ReVersionRegularDatasetnoRandom, 
                             ReVersionandRegularDataset, ReVersionandRegularDatasetv2, 
                             ReVersionandRegularDatasetv3,ReVersionandRegularDatasetv3forencoder)
from dp_dataset import DPDatasetnoRegular, DPDatasetv2forencoder
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
from utils import encoder
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
        default='warp',
        help="mode to get the initial latent noise, such as random, DDIM, encoder, or else.",
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
        default=9000,
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

def collate_fn(examples, dp = False):
    latent_embs = [example["latent_emb"] for example in examples]
    names = [example["name"] for example in examples]
    original_sizes = [example["original_sizes"] for example in examples]
    # crop_top_lefts = [example["crop_top_left"] for example in examples]
    bboxes = [example["bboxes"] for example in examples]


    latent_embs = torch.stack(latent_embs)
    latent_embs = latent_embs.to(memory_format=torch.contiguous_format).float()

    batch = {
            "latent_embs": latent_embs,
            "name": names,
            "original_sizes": original_sizes,
            # "crop_top_lefts": crop_top_lefts,
            "bboxes": bboxes,
        }
    return batch

def main():

    args = parse_args()

    from templates.relation_words import relation_words


    args.learning_rate = args.noise_learning_rate
    args.denoise_loss_weight = args.noise_loss_weight

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
    tokenizer = None
    if args.dp:
        train_dataset = DPDatasetv2forencoder(
            data_root=args.train_data_dir,
            size=args.resolution,
            repeats=args.repeats,
            crop_p=args.crop_p,
            flip_p=args.flip_p,
            set="train",)
    else:
        # Dataset and DataLoaders creation:
        train_dataset = ReVersionandRegularDatasetv3forencoder(
            data_root=args.train_data_dir,
            size=args.resolution,
            repeats=args.repeats,
            crop_p=args.crop_p,
            flip_p=args.flip_p,
            set="train",
            )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, dp=args.dp),
        num_workers=args.dataloader_num_workers)

    

    
    dataset_length = train_dataset.num_images

    local_f = open(os.path.join(args.train_data_dir, 'text.json'))
    templates = json.load(local_f, object_pairs_hook=OrderedDict)
    noise_dict = {}
    for index, (key, value) in enumerate(templates.items()):
        noise_dict[key] = index
    
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

    
    if args.dp:
        num_classes = -1
        label_maps = []
        cls_f = open(os.path.join(args.train_data_dir, 'cls.json'))
        clses = json.load(cls_f, object_pairs_hook=OrderedDict)
        for index, (key, value) in enumerate(clses.items()):
            num_classes = max(num_classes, value)
        num_classes += 1
        for i in range(num_classes):
            semantic_labels = torch.ones((args.resolution//8, args.resolution//8)).to(accelerator.device)

            # 转换为one-hot编码
            one_hot = F.one_hot(semantic_labels.long(), num_classes=num_classes)  # 形状为 (H, W, 5)

            # 将one-hot编码转换为 (5, H, W) 的形状
            one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)  # 形状为 (1, 5, H, W)
            label_maps.append(one_hot)





    if args.noise_mode == 'encoder':
        position_encoder = encoder.PositionalEncoder(d_input= 4, 
                                                    n_freqs = 10,
                                                    log_space = False)
        noise_embedding_encoder = encoder.LatentEncoder(d_input = position_encoder.d_output)
        # torch.save(noise_embedding_encoder,'./{}/{}/{}_noisencoder_ori.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))

        params_to_optimize = [{
                    "params": noise_embedding_encoder.parameters(),
                    "lr": args.noise_learning_rate,
                }]
        
    elif args.noise_mode == 'warp':
        # position_encoder = encoder.PositionalEncoder(d_input= 4, 
        #                                             n_freqs = 10,
        #                                             log_space = False)
        noise_embedding_encoder = encoder.Warpper() if not args.dp else encoder.DPWarpper(num_classes)
        # torch.save(noise_embedding_encoder,'./{}/{}/{}_noisewarpper_ori.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))

        params_to_optimize = [{
                    "params": noise_embedding_encoder.parameters(),
                    "lr": args.noise_learning_rate,
                }]
        

    

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
    # text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     text_encoder, optimizer, train_dataloader, lr_scheduler)
    if args.noise_mode == 'encoder' or args.noise_mode == 'warp': 
        noise_embedding_encoder = accelerator.prepare(noise_embedding_encoder)
    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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
    # orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()


    method =  'noise_new_encoder_0401_'+args.train_data_dir.split('/')[-1]
    out_root = '{}/lr{}'.format(method, args.noise_learning_rate) 
    sub_root = '{}_crop{}_flip{}/'.format(args.train_data_dir.split('/')[-1], str(args.crop_p), str(args.flip_p))
    if not os.path.exists(os.path.join(out_root, sub_root)): os.makedirs(os.path.join(out_root, sub_root))
    
    if args.noise_mode == 'encoder':
        torch.save(noise_embedding_encoder,'./{}/{}/{}_noisencoder_ori.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))
    elif args.noise_mode == 'warp':
        torch.save(noise_embedding_encoder,'./{}/{}/{}_noisewarpper_ori.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))

        

    noise_embedding_encoder = noise_embedding_encoder.requires_grad_()
    

    # noise_embedding_weight = noise_embedding_weight.requires_grad_()
    for epoch in range(first_epoch, args.num_train_epochs):

        if epoch == args.num_train_epochs // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                
        noise_embedding_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # if 'hug' in args.train_data_dir.split('/')[-1]: continue
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(noise_embedding_encoder):
                
                loss = 0.0
                gt = batch["latent_embs"].to(accelerator.device, dtype=weight_dtype)
                bsz = batch["latent_embs"].shape[0]
                # L_denoise
                input_noise = torch.randn_like(gt)
                mask_set = []
                target_set = []
                for b in range(bsz):
                    name_b = batch["name"][b]
                    h, w = batch["original_sizes"][b][0], batch["original_sizes"][b][1]
                    if args.dp:
                        mask = label_maps[batch["cls"][b]].clone()
                        mask_set.append(mask)

                        # input_noise = torch.randn_like(gt[b:b+1])
                        # input_warp = torch.cat([mask, input_noise], 1)
                        # output_warp = noise_embedding_encoder(input_warp)

                        target_set.append(gt[b:b+1])
                    else:
                        for bbox in batch["bboxes"][b]:
                            min_height, min_width, max_height, max_width = bbox
                            # min_height = torch.round(args.resolution/h*min_height)
                            # max_height = torch.round(args.resolution/h*max_height)
                            # min_width = torch.round(args.resolution/w*min_width)
                            # max_width = torch.round(args.resolution/w*max_width)
                            if args.noise_mode == 'encoder':
                                pos_input = torch.Tensor([min_height, min_width, max_height, max_width]).to(accelerator.device, dtype=weight_dtype)
                                norm_pos_input = pos_input / args.resolution
                                pos_emb = position_encoder(norm_pos_input)
                                noise_enc = noise_embedding_encoder(pos_emb)
                                nh, nw = max_height//8 - min_height//8, max_width//8 - min_width//8
                                mode = 'area' if min(nh,nw) <= noise_enc.shape[3]  else 'bicubic'
                                resize_noise = torch.nn.functional.interpolate(noise_enc, size = [nh, nw], mode = mode) # shape: 1,4,64,64
                                
                                tar_noise = gt[b:b+1].detach()
                                target = tar_noise[0:1,:,min_height//8:max_height//8, min_width//8:max_width//8]
                                encoder_loss = F.mse_loss( resize_noise.float(), target.float(), reduction="mean")
                            elif args.noise_mode == 'warp':
                                mask = torch.zeros((1,1,gt.shape[2], gt.shape[3])).to(accelerator.device, dtype=weight_dtype)
                                mask[0:1,:,min_height//8:max_height//8, min_width//8:max_width//8] = 1
                                mask_set.append(mask)

                                # input_noise = torch.randn_like(gt[b:b+1])
                                # input_warp = torch.cat([mask, input_noise], 1)
                                # output_warp = noise_embedding_encoder(input_warp)

                                target_noise = input_noise[b:b+1].detach().clone()

                                target_noise[0:1,:,min_height//8:max_height//8, min_width//8:max_width//8] = gt[b:b+1,:,min_height//8:max_height//8, min_width//8:max_width//8]
                                target_set.append(target_noise)
                                    # encoder_loss = F.mse_loss( output_warp.float(), (target_noise - input_noise).float(), reduction="mean")
                masks = torch.cat(mask_set, 0)
                input_warps = torch.cat([masks, input_noise], 1)

                targets = torch.cat(target_set, 0)
                output_warp = noise_embedding_encoder(input_warps)
                if args.dp:
                    encoder_loss = F.mse_loss( output_warp.float(), (targets-input_noise).float(), reduction="mean")
                else:
                    encoder_loss = F.mse_loss( (output_warp*masks).float(), ((targets-input_noise)*masks).float(), reduction="mean")
                weighted_denoise_loss = args.denoise_loss_weight * encoder_loss
                loss += weighted_denoise_loss

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(
                        args.output_dir,
                        f"learned_embeds-steps-{global_step}.bin")
                    if args.noise_mode == 'encoder':
                        torch.save(noise_embedding_encoder,'./{}/{}_noisencoder.pth'.format(args.output_dir, args.train_data_dir.split('/')[-1]))
                    elif args.noise_mode == 'warp':
                        torch.save(noise_embedding_encoder,'./{}/{}_noisewarpper.pth'.format(args.output_dir, args.train_data_dir.split('/')[-1]))

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


            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # validation
      
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:


        if args.noise_mode == 'encoder':
            torch.save(noise_embedding_encoder,'./{}/{}/{}_noisencoder.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))
        elif args.noise_mode == 'warp':
            torch.save(noise_embedding_encoder,'./{}/{}/{}_noisewarpper.pth'.format(out_root, sub_root, args.train_data_dir.split('/')[-1]))
        

    accelerator.end_training()


if __name__ == "__main__":
    main()
