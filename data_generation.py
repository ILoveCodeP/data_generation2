import os
import transformers
from axolotl.common.cli import TrainerCliArgs
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from axolotl.cli import (
    load_cfg,
    load_datasets,
)
import torch
from torch.utils.data import Dataset, DataLoader
from LinearModel import Classifier
import time
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# extract assistant contents
def get_non_negative_indices(lst):
    return [i for i, num in enumerate(lst) if num != -100]

def custom_binary_cross_entropy_loss(output, label):
    probas = F.softmax(output, dim=-1)
    loss = 10*label * torch.log(probas[:,0]) + (1 - label) * torch.log(probas[:,1])
    loss = -torch.mean(loss)
    return loss

def load_dataset():
    parsed_cfg = load_cfg(config='/AI_home/lijipeng/Medusa/axolotl/examples/medusa/llama_chat_13b.yml')

    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
    dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)

    train_dataset = dataset_meta.train_dataset
    return train_dataset


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)  # 标签通常为长整型
        return feature_tensor, label_tensor
    
def evaluation(model,eval_file_path_data, eval_file_path_label):
    with open(eval_file_path_data, 'rb') as file:
        eval_data = pickle.load(file)
    with open(eval_file_path_label, 'rb') as file:
        eval_label = pickle.load(file)

    batch_size = 1024  # 你可以根据需要设置批次大小
    eval_dataset = CustomDataset(eval_data, eval_label)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    losses = []
    preds = []
    for features,labels in dataloader:
        features = features.cuda()
        labels = labels.cuda()
        output = model(features)
        loss = custom_binary_cross_entropy_loss(output, labels)
        losses.append(loss.item())
    return np.mean(losses)
    

def main():
    tensorboard_log_dir = f"/AI_home/lijipeng/logs/all_data/test_all_dataset"
    if not os.path.exists(tensorboard_log_dir):
        # 创建目录
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    # load llm model 
    model_path1 = '/AI_home/lijipeng/llama/Llama-2-7b-chat-hf'
    model_path2 = '/AI_home/lijipeng/llama/Llama-2-13b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_path1,use_fast=False,add_bos_token=False, model_max_length=4096,padding_side="right",trust_remote_code=True)
    model1 = AutoModelForCausalLM.from_pretrained(model_path1,  torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True,output_hidden_states=True,use_flash_attention_2=True).eval()
    model2 = AutoModelForCausalLM.from_pretrained(model_path2,  torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True,output_hidden_states=True,use_flash_attention_2=True).eval()
    train_dataset = load_dataset()
    # load classifier
    model = Classifier(hidden_size=4096,out_dim=2, res_layers=1)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.01)
    iteration = 0
    # eval_file_path_data = 'eval_data_lists.pickle'
    # eval_file_path_label = 'labels_lists.pickle'

    rs_labels_lists = []
    training_data_lists = []

    for ind in range(len(train_dataset)):

        print("ind is", ind)
        train_dict = train_dataset[ind]

        # extract assistant tokens
        index = get_non_negative_indices(train_dict['labels'])
        modified_index = [i - 1 for i in index]

        with torch.no_grad():
            outputs1 = model1(torch.tensor(train_dict['input_ids']).unsqueeze(0).cuda())
            outputs2 = model2(torch.tensor(train_dict['input_ids']).unsqueeze(0).cuda())
        
            logits1 = outputs1.logits[:, modified_index, :]
            logits2 = outputs2.logits[:, modified_index, :]
        
            softmax_tensor1 = F.softmax(logits1, dim=2)
            softmax_tensor2 = F.softmax(logits2, dim=2)

            # 获取每个位置的最大索引
            max_indices1 = torch.argmax(softmax_tensor1, dim=2)  
            max_indices2 = torch.argmax(softmax_tensor2, dim=2)

            # 比较两个张量的最大索引是否相等
            equal_indices = torch.eq(max_indices1, max_indices2)
            # 将 True/False 值转换为 0/1
            rs_labels = (~equal_indices).int()

            rs_labels_list = rs_labels.tolist()

            rs_labels_lists.extend(rs_labels_list[0])

            training_data = outputs1.hidden_states[-1][:, modified_index, :][0]

            training_data_lists.extend(training_data)
        if ind % 10 == 0:
            model.train()
            print('start training classifier')
            batch_size = 1024  
            custom_dataset = CustomDataset(training_data_lists, rs_labels_lists)
            dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
            training_data_lists = []
            rs_labels_lists = []

            for epoch in range(1):
                for features, labels in dataloader:
                    features = features.cuda()
                    print(features.shape)
                    labels = labels.cuda()
                    print(labels.shape)
                    output = model(features)
                    loss = custom_binary_cross_entropy_loss(output, labels)

                    writer.add_scalar('training loss', loss, iteration)
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    iteration += 1
                    print('loss is',loss)
            torch.save(model.state_dict(), '/AI_home/lijipeng/fix_classifier/original.pth')
        # if ind % 1000 == 0:
        #     eval_loss = evaluation(model,eval_file_path_data, eval_file_path_label)
        #     writer.add_scalar('evaluation loss', eval_loss, iteration)
if __name__ == "__main__":
    main()