import os
import argparse

import torch, cv2
from PIL import Image

from diffusers import StableDiffusionPipeline
from templates.templates import inference_templates

from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
# import ptp_utils
# import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from utils import ptp_utils
import math
from lora import (
    save_lora_weight,
    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    get_target_module,
    save_lora_layername,
    monkeypatch_or_replace_lora,
    monkeypatch_remove_lora,
    set_lora_requires_grad,
    tune_lora_scale
)

"""
Inference script for generating batch results
"""

LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        if Image.open(image_path).mode == 'L':
            image = np.expand_dims(np.array(Image.open(image_path)),2).repeat(3,2)
        else:
            image = np.array(Image.open(image_path))[:, :, :3]
        # 
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512))).astype(np.float32)
    return image


class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            if image.shape[0] == 1: 
                image = image[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, steps = NUM_DDIM_STEPS):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, steps = NUM_DDIM_STEPS):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent, steps)
        return image_rec, ddim_latents

    


    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        # ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
    
    @torch.no_grad()
    def ddim_loop_complete(self, image, steps = NUM_DDIM_STEPS):
        latent = self.image2latent(image)
        all_latent = [latent]
        latent_cur = latent.clone().detach()
        for i in range(steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            latent_cur = self.get_noise_pred(latent_cur, t, True, self.context)
            all_latent.append(latent_cur)
        return all_latent, self.latent2image(latent_cur)

    @torch.no_grad()
    def ddpm_loop_complete(self, image, steps = NUM_DDIM_STEPS):
        latent = self.image2latent(image)
        all_latent = [latent]
        latent_cur = latent.clone().detach()
        for i in range(steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise = torch.randn_like(latent_cur)
            # latent_cur = self.get_noise_pred(latent_cur, t, True, self.context)
            latent_cur = self.next_step(noise, t, latent_cur)
            all_latent.append(latent_cur)
        return all_latent, self.latent2image(latent_cur)

    @torch.no_grad()
    def ddim_redenoise(self, latents, steps = NUM_DDIM_STEPS):
        latent_cur = latents[-1]
        for i in range(steps):
            t = self.model.scheduler.timesteps[NUM_DDIM_STEPS - steps + i]
            latent_cur = self.get_noise_pred(latent_cur, t, False, self.context)
        image_redenoi = self.latent2image(latent_cur)
        return image_redenoi, latent_cur

    def noise_inference(self, latents, prompt: str,strength = 1):
        self.init_prompt(prompt)
        image_inver, latent_inver = self.ddim_redenoise([latents], int(strength * NUM_DDIM_STEPS))
        return [image_inver]
    
    @torch.no_grad()
    def init_p2p_prompt(self, prompt: str,controller = None, uncond_embeddings = None):
        batch_size = len(prompt)
        ptp_utils.register_attention_control(self.model, controller)
        img_height = img_width = 512
        
        if '<R>' in prompt[1]:
            relation_token_id = self.tokenizer.encode('<R>', truncation=False, add_special_tokens=False)[0]
            self.relation_index = [controller.encode_prompts[0][1].index(relation_token_id)]
        else:
            self.relation_index = None
        if controller.key_words is not None:
            self.key_prompt_indexes = []
            for word in  controller.key_words:
                key_token_id = self.tokenizer.encode(word, truncation=False, add_special_tokens=False)[0]
                for rindex, rid in enumerate(controller.encode_prompts[0][1]):
                    if rid == key_token_id:
                        self.key_prompt_indexes.append(rindex)
        height, width = img_height//8, img_width//8
        res = [[math.ceil(height/2) , math.ceil(width/2)],[math.ceil(height/4) , math.ceil(width/4)],[math.ceil(height/8) , math.ceil(width/8)]]
        self.num_pixels = [res[0][0] * res[0][1], res[1][0] * res[1][1], res[2][0] * res[2][1]]
        self.res = res
        
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # print(text_input)
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        max_length = text_input.input_ids.shape[-1]

        if uncond_embeddings is None:
            uncond_input = self.model.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings_ = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        else:
            uncond_embeddings_ = None
        # if latent.shape[0]<batch_size:
        #     latent, latents = ptp_utils.init_latent(latent, self.model, height, width, None, batch_size)
        # else:
        #     latents = latent
        
        self.context = torch.cat([uncond_embeddings_, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def noise_inference_p2p_guidance(self, latents, prompts: str,controller = None,strength = 1, guidance = True, offset_noises = None):

        self.init_p2p_prompt(prompts, controller = controller)
        # ptp_utils.register_attention_control(self.model, controller)
        # image_inver, latent_inver = self.ddim_redenoise([latents], int(strength * NUM_DDIM_STEPS))

        latent_cur = latents
        steps = int(strength * NUM_DDIM_STEPS)
        for i in range(steps):
            t = self.model.scheduler.timesteps[NUM_DDIM_STEPS - steps + i]
            latent_cur = self.get_noise_pred_p2p_guidance(latent_cur, t, False, self.context, guidance = guidance, offset_noises = offset_noises)
            controller.reset_layer_num()
        image_inver = self.latent2image(latent_cur)
        return [image_inver]
    
    def get_noise_pred_offset(self, latents, t, offset_noises, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if offset_noises is not None:
            for offset in offset_noises:
                min_height, min_width, max_height, max_width = offset['bbox']
                noise_pred[:,:,min_height:max_height, min_width:max_width] -= offset['scale'] * offset['noise'][:,:,min_height:max_height, min_width:max_width]
        # _ = self.model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
    
    @torch.no_grad()
    def ddim_redenoise_offset(self, latents, offset_noises, steps = NUM_DDIM_STEPS):
        latent_cur = latents[-1]
        for i in range(steps):
            t = self.model.scheduler.timesteps[NUM_DDIM_STEPS - steps + i]
            latent_cur = self.get_noise_pred_offset(latent_cur, t, offset_noises, False, self.context)
        image_redenoi = self.latent2image(latent_cur)
        return image_redenoi, latent_cur


    def noise_inference_offset(self, latents, prompt: str, offset_noise, strength = 1):
        self.init_prompt(prompt)
        image_inver, latent_inver = self.ddim_redenoise_offset([latents], offset_noise, int(strength * NUM_DDIM_STEPS))
        return [image_inver]
    

    def halfinvert(self, image_path: str, prompt: str, offsets=(0,0,0,0), verbose=False, strength = 1, mode = 'ddpm'):
        self.init_prompt(prompt)
        # ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        if mode == 'ddpm':
            ddim_latents, image_noise = self.ddpm_loop_complete(image_gt, int(strength * NUM_DDIM_STEPS)) 
        else:
            ddim_latents, image_noise = self.ddim_loop_complete(image_gt, int(strength * NUM_DDIM_STEPS)) 
        if verbose:
            print("DDIM Redenoising...")
        image_inver, latent_inver = self.ddim_redenoise(ddim_latents, int(strength * NUM_DDIM_STEPS))
        return (image_gt, image_inver, image_noise), ddim_latents

    def gen_noise(self, image_path: str, prompt: str, offsets=(0,0,0,0), verbose=False, strength = 1, mode = 'ddpm'):
        self.init_prompt(prompt)
        # ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if mode == 'ddpm':
            ddim_latents, image_noise = self.ddpm_loop_complete(image_gt, int(strength * NUM_DDIM_STEPS)) 
        else:
            ddim_latents, image_noise = self.ddim_loop_complete(image_gt, int(strength * NUM_DDIM_STEPS)) 

        return (image_gt, image_noise), ddim_latents
    
    

    def __init__(self, model, guidance_weights = 1):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        
        self.tokenizer = self.model.tokenizer
        
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.guidance_weights = guidance_weights


