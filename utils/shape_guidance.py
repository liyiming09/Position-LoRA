import torch
from torch import tensor
from einops import rearrange


#### Self guidance equations

def normalise(x): return (x - x.min()) / (x.max() - x.min())

def threshold_attention(attn, s=10):
    norm_attn = s * (normalise(attn) - 0.5)
    return normalise(norm_attn.sigmoid())

def get_shape(attn, s=20): return threshold_attention(attn, s)
def get_size(attn): return threshold_attention(attn).sum((0,1)).mean() #1/attn.shape[-2] * threshold_attention(attn).sum((1,2)).mean()
def get_centroid(attn):
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    hs = torch.arange(h).view(-1, 1, 1).to(attn.device)
    ws = torch.arange(w).view(1, -1, 1).to(attn.device)
    attn = rearrange(attn.mean(0), '(h w) d -> h w d', h=h)
    weighted_w = torch.sum(ws * attn, dim=[0,1])
    weighted_h = torch.sum(hs * attn, dim=[0,1])
    return torch.stack([weighted_w, weighted_h]) / attn.sum((0,1))
def get_appearance(attn, feats):
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    shape = get_shape(attn).detach().mean(0).view(h,w,attn.shape[-1])
    feats = feats.mean((0,1))[:,:,None]
    return (shape*feats).sum() / shape.sum()

#### G functions

#Single image editing. These are the functions that are closest to the paper.
#  In the experiments section below,
#  I played around with variations on these equations in pursuit of better results.

def fix_shapes(orig_attns, edit_attns, tau=1):
    shapes = []
    # longer_attns = edit_attns if len(edit_attns) >= len(orig_attns) else orig_attns
    # shorter_attns = orig_attns if len(edit_attns) >= len(orig_attns) else edit_attns
    shorter_nums = min(len(edit_attns), len(orig_attns))
    for i in range(shorter_nums):
        orig, edit = orig_attns[i], edit_attns[i]
        delta = tau*get_shape(orig) - get_shape(edit)
        shapes.append(delta.mean())
    return torch.stack(shapes).mean()

def fix_appearances(orig_attns, orig_feats, edit_attns, edit_feats, attn_idx=-1):
    appearances = []
    shorter_nums = min(len(edit_attns), len(orig_attns))
    for i in range(shorter_nums):
        orig, edit = orig_attns[i], edit_attns[i]
        appearances.append((get_appearance(orig, orig_feats) - get_appearance(edit, edit_feats)).pow(2).mean())
    return torch.stack(appearances).mean()

def fix_sizes(orig_attns, edit_attns, tau=1):
    sizes = []
    shorter_nums = min(len(edit_attns), len(orig_attns))
    for i in range(shorter_nums):
        orig, edit = orig_attns[i], edit_attns[i]
        sizes.append(tau*get_size(orig) - get_size(edit))
    return torch.stack(sizes).mean()

def position_deltas(orig_attns, edit_attns, delta_centroid = None, target_centroid=None):
    positions = []
    shorter_nums = min(len(edit_attns), len(orig_attns))
    # for i in range(shorter_nums):
    # only move the region 2 with a BlendMode:AddRelation
    orig, edit = orig_attns[2], edit_attns[2]
    if delta_centroid is not None:
        target = tensor(delta_centroid).to(orig.device) + get_centroid(orig)
    elif target_centroid is not None:
        target = tensor(target_centroid).to(orig.device)
    else:
        target = get_centroid(orig)
    positions.append(target.to(orig.device) - get_centroid(edit))
    return torch.stack(positions).mean()


def fix_selfs(origs, edits):
    shapes = []
    for i in range(len(edits)):
        shapes.append((threshold_attention(origs[i]) - threshold_attention(edits[i])).mean())
    return torch.stack(shapes).mean()

def get_attns(attn_storage, attn_type='attn2'):
    origs = attn_storage[0]#[v['orig'] for k,v in attn_storage.storage.items() if attn_type in k]
    edits = attn_storage[1]#[v['edit'] for k,v in attn_storage.storage.items() if attn_type in k]
    return origs, edits

def edit_appearance(attn_storage, appearance_weight=0.5, orig_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    return appearance_weight*fix_appearances(origs, orig_feats, edits, edit_feats, **kwargs)

def edit_layout(attn_storage, shape_weight=1, **kwargs):
    origs, edits = get_attns(attn_storage)
    return shape_weight*fix_shapes(origs, edits)

def resize_object(attn_storage, relative_size=2, shape_weight=1, size_weight=1, appearance_weight=0.1, orig_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    # if len(indices) > 1: 
    #     obj_idx, other_idx = indices
    #     indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight*fix_shapes(origs, edits)
    # appearance_term = appearance_weight*fix_appearances(origs, orig_feats, edits, edit_feats)
    size_term = size_weight*fix_sizes(origs, edits, tau=relative_size)
    return shape_term  + size_term #+ appearance_term

def move_object(attn_storage, delta_centroid = None, target_centroid=None, shape_weight=1, size_weight=1, appearance_weight=0.5, position_weight=1, orig_feats=None, edit_feats=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    # if len(indices) > 1: 
    #     obj_idx, other_idx = indices
    #     indices = torch.cat([obj_idx, other_idx])
    shape_term = shape_weight*fix_shapes(origs, edits)
    # appearance_term = appearance_weight*fix_appearances(origs, orig_feats, edits, edit_feats)
    size_term = size_weight*fix_sizes(origs, edits)
    position_term = position_weight*position_deltas(origs, edits, delta_centroid = delta_centroid, target_centroid=target_centroid)
    return shape_term + size_term + position_term#+ appearance_term
