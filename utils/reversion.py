
import torch, math
import torch.nn.functional as F
from attn_guidance import (AttnStorage, CustomAttnProcessor, prepare_attention, get_attn_dict, get_attns)

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
  
IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif')


def is_image_file(filename):
    # return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    return filename.endswith(IMG_EXTENSIONS)
