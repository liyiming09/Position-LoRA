from diffusers.models.attention_processor import AttnProcessor, Attention
import math, random, torch
import fastcore.all as fc

def get_attn_dict(processor, model):
    attn_procs = {}
    for name in model.attn_processors.keys():
        attn_procs[name] = processor(name=name)
    return attn_procs

def prepare_attention(model, attn_storage, pred_type='orig', set_store=True):
    for name, module in model.attn_processors.items(): module.set_storage(set_store, pred_type)

def get_attns(attn_storage, attn_type='attn2'):
    origs = [v['orig'] for k,v in attn_storage.storage.items() if attn_type in k]
    # edits = [v['edit'] for k,v in attn_storage.storage.items() if attn_type in k]
    return origs#, edits
    
class AttnStorage:
    def __init__(self): self.storage = {}
    def __call__(self, attention_map, name, pred_type='orig'): 
        if not name in self.storage: self.storage[name] = {}
        self.storage[name][pred_type] = attention_map
    def flush(self): self.storage = {}

class CustomAttnProcessor(AttnProcessor):
    def __init__(self, attn_storage, name=None): 
        fc.store_attr()
        self.store = False
        self.type = "attn2" if "attn2" in name else "attn1"
    def set_storage(self, store, pred_type): 
        self.store = store
        self.pred_type = pred_type
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
     
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs.requires_grad_(True)
        
        if self.store: self.attn_storage(attention_probs, self.name, pred_type=self.pred_type) ## stores the attention maps in attn_storage
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states

def prepare_attention(model, attn_storage, pred_type='orig', set_store=True):
    for name, module in model.attn_processors.items(): module.set_storage(set_store, pred_type)
