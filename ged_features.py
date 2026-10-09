"""Learner-English grammatical-error features and transcript embeddings."""

import re
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.cache_meta import write_meta
from src.sentences import split_sentences

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL='rahuln2002/roberta-base-20k-GED'
parser=argparse.ArgumentParser()
parser.add_argument('--features',type=Path,default=Path('cache/features.csv'))
parser.add_argument('--ged',type=Path,default=Path('cache/ged_features.csv'))
parser.add_argument('--embeddings',type=Path,default=Path('cache/ged_embeddings.npz'))
args=parser.parse_args()
torch.set_num_threads(4)
tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,use_safetensors=True).eval()
frame=pd.read_csv(args.features)

owners=[];sentences=[];lengths=[]
for i,text in enumerate(frame.text.fillna('')):
 for part in split_sentences(text):
  owners.append(i);sentences.append(part);lengths.append(len(re.findall(r"[A-Za-z']+",part)))

print('sentences',len(sentences),flush=True)
scores=np.zeros(len(sentences))
with torch.inference_mode():
 for start in range(0,len(sentences),16):
  batch=tokenizer(sentences[start:start+16],padding=True,truncation=True,max_length=128,return_tensors='pt')
  scores[start:start+16]=model(**batch).logits.softmax(-1)[:,1].numpy()
  if start%512==0:print('sentences',start,'/',len(sentences),flush=True)

groups=[[] for _ in range(len(frame))]
for owner,score,length in zip(owners,scores,lengths):groups[owner].append((score,length))
rows=[]
for items in groups:
 if items:
  values=np.array([v for v,_ in items]);weights=np.array([max(n,1) for _,n in items])
  rows.append({'ged_mean':values.mean(),'ged_min':values.min(),'ged_max':values.max(),'ged_median':np.median(values),'ged_std':values.std(),'ged_weighted':np.average(values,weights=weights),'ged_high_frac':np.mean(values>.5),'ged_very_high_frac':np.mean(values>.8)})
 else:
  rows.append(dict.fromkeys(['ged_mean','ged_min','ged_max','ged_median','ged_std','ged_weighted','ged_high_frac','ged_very_high_frac'],0.0))
out=pd.DataFrame(rows);out.insert(0,'filename',frame.filename);out.insert(0,'split',frame.split)
args.ged.parent.mkdir(parents=True,exist_ok=True)
out.to_csv(args.ged,index=False)

texts=[t if t.strip() else '.' for t in frame.text.fillna('')]
vectors=[]
with torch.inference_mode():
 for start in range(0,len(texts),8):
  batch=tokenizer(texts[start:start+8],padding=True,truncation=True,max_length=256,return_tensors='pt')
  hidden=model.roberta(**batch).last_hidden_state
  mask=batch['attention_mask'].unsqueeze(-1)
  pooled=(hidden*mask).sum(1)/mask.sum(1)
  vectors.append(pooled.numpy())
  if start%80==0:print('transcripts',start,'/',len(texts),flush=True)
embedding=np.vstack(vectors).astype(np.float32)
args.embeddings.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.embeddings,split=frame.split.to_numpy(str),filename=frame.filename.to_numpy(str),embedding=embedding)
write_meta(args.ged, row_count=len(out))
print('saved',embedding.shape,flush=True)
