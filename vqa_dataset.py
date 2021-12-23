# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         vqa_dataset
# Description:  VQAdataset : vision{maml + autoencoder} & questions & labels
# Author:       Boliu.Kelvin
# Date:         2020/4/6
#-------------------------------------------------------------------------------

import _pickle as cPickle
import numpy as np
from torch.utils.data import Dataset
import os
import utils
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer

class VQAFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data', question_len=12):
        super(VQAFeatureDataset, self).__init__()
        self.args = args
        assert name in ['train', 'validate','test']
        ans2label_path = os.path.join(dataroot,  'ans2label.pkl')
        print(ans2label_path)
        label2ans_path = os.path.join(dataroot,  'label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        # Get the word dictionary
        self.dictionary = dictionary
        #Get the target [question,image_name,labels,scores] of [trian or validate]
        self.entries = cPickle.load(open(os.path.join(dataroot,name+'_target.pkl'), 'rb'))

        if name =='train':
            self.images_path = os.path.join(dataroot, 'TrainVal-Sets/TrainingSet/Train_images')
        elif name =='validate':
            self.images_path = os.path.join(dataroot, 'TrainVal-Sets/ValidationSets/Validation-Images')
        elif name =='test':
            self.images_path = os.path.join(dataroot, 'TrainVal-Sets/TestSets/TestSet-Images/VQA-500-Images')
        self.tokenize(question_len)
        # self.tensorize()
        if args.autoencoder and args.maml:
            self.v_dim = args.v_dim *2
        if args.other_model:
            # self.v_dim = args.v_dim *2
            self.v_dim = 1984

        self.tokenizer = AutoTokenizer.from_pretrained("./data/emilyalsentzer/Bio_ClinicalBERT")
        self.max_len = question_len

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens


    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        image_name = entry['image_name']
        question_title = entry['question']
        question_title = " ".join(question_title.split())

        inputs_ques = self.tokenizer.encode_plus(
            question_title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        image_data = None

        compose = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224,scale=(0.95,1.05),ratio=(0.95,1.05)),
        transforms.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                     np.array([63.0, 62.1, 66.7]) / 255.0)

        ])
        other_images_data = Image.open(os.path.join(self.images_path, image_name) + '.jpg')
        image_data = compose(other_images_data)


        labels = np.array(entry['labels'])


        scores = np.array(entry['scores'],dtype=np.float32)

        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, torch.tensor(labels), torch.tensor(scores))


        return  image_data,inputs_ques,target,image_name



    def __len__(self):
        return len(self.entries)
