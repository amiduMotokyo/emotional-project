"""Attachment 2/4 supplied 768-dimensional features; shared mask/scaling schema."""
import numpy as np
import torch
from B.src.data import assemble_sample,fit_audio_vision_scale,apply_audio_vision_scale

def prepare(raw):
    text=np.nan_to_num(np.asarray(raw['text'],dtype=np.float32))
    bert=np.asarray(raw['text_bert']);audio=np.asarray(raw['audio']);vision=np.asarray(raw['vision'])
    if text.ndim==2:text,bert,audio,vision=text[None],bert[None],audio[None],vision[None]
    if text.shape[1:]!=(50,768):raise ValueError('Expected supplied aligned 50 x 768 text')
    d=assemble_sample(bert,text,audio,vision,raw.get('classification_labels'),raw.get('regression_labels'))
    d['tmask']&=d['token_ids']!=0
    d['sample_id']=np.asarray(raw['id'] if isinstance(raw['id'],(list,np.ndarray)) else [str(raw['id'])]).astype(str)
    return d

def subset(d,indices):return {k:v[indices].copy() for k,v in d.items()}

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self,d):self.d=d
    def __len__(self):return len(self.d['text'])
    def __getitem__(self,i):
        return tuple(torch.from_numpy(self.d[k][i].copy()) for k in ['text','audio','vision','tmask','amask','vmask'])+(torch.tensor(int(self.d['cls'][i])),torch.tensor(float(self.d['score'][i]),dtype=torch.float32))
