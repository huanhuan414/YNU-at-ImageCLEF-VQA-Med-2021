import torch
import torch.nn as nn
from FixVgg16 import FixVgg16
from BioBert import BERTClass
from transformers import AutoTokenizer
from classifier import SimpleClassifier
from mfb import CoAtt

class BAN_Model(nn.Module):
    def __init__(self, dataset,args):
        super(BAN_Model, self).__init__()
        self.args = args
        self.unet = FixVgg16()
        self.bio_bert = BERTClass()
        self.mfh = CoAtt()
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(2000, dataset.num_ans_candidates)
    def forward(self,v,q):
        v_emb = self.unet(v)  #input: b,c,h,w c==3 ; output= b,1,1,1984
        v_emb = v_emb.squeeze(2) # b,1,1472

        q_emb = self.bio_bert(q['input_ids'].to(self.args.device).squeeze(1),q['attention_mask'].to(self.args.device).squeeze(1),q['token_type_ids'].to(self.args.device).squeeze(1))

        q_emb = q_emb.unsqueeze(1)  #b  768

        fuse_feature = self.mfh(v_emb,q_emb)

        fuse_feature = self.drop(fuse_feature)

        out_feature = self.classifier(fuse_feature)

        return out_feature


