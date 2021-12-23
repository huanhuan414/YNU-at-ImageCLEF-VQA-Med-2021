import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, AutoModel
import transformers

class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.AutoModel.from_pretrained("./data/emilyalsentzer/Bio_ClinicalBERT")
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 1024)

    def forward(self, ids, mask, token_type_ids):
        output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # output = self.l2(output_1[1])
        # output = self.l3(output_2)
        return output_1[1]

if __name__=="__main__":
    model = BERTClass()
    print(model)
