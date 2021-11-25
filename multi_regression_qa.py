import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
is_train = True #True: 训练模型，False:加载模型进行预测
uses_full_data = False  #全部数据，含val_data
model_path = 'multi_regression_model_qa.pkl'
max_epoch = 3
path = os.getcwd()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

train = pd.read_csv('train_dataset.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)
print(train.shape)
print(train.head())

test = pd.read_csv('test_dataset.tsv', sep='\t')
print(test.shape)
print(test.head())


submit = pd.read_csv('submit_example.tsv', sep='\t')
print(submit.shape)
print(submit.head())

train = train[train['emotions'].notna()]
print(train.shape)


train['character'].fillna('无角色', inplace=True)
test['character'].fillna('无角色', inplace=True)

train['text'] = train['content'].astype(str)  # + ' 角色: ' + train['character'].astype(str)
test['text'] = test['content'].astype(str)   #+ ' 角色: ' + test['character'].astype(str)
train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])

full_data_shuff = train.sample(frac=1, random_state=42)
split_pos = int(len(full_data_shuff)*0.8)
train_data = full_data_shuff if uses_full_data else full_data_shuff.iloc[:split_pos,:]
val_data = full_data_shuff.iloc[split_pos:,:]
class MyDataSet(Dataset):
    def __init__(self,texts,characters,labels=None):
        self.texts = list(texts)
        self.characters = list(characters)
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = None



    def __getitem__(self, index):
        if self.labels is not None:
            return [self.texts[index],self.characters[index],self.labels[index]]
        else:
            return [self.texts[index],self.characters[index]]

    def __len__(self):
        return len(self.texts)
train_dataset = MyDataSet(train_data['text'].values,characters=train_data['character'].values,
                          labels=train_data['labels'].values)
train_data_loader = DataLoader(train_dataset,batch_size=32,shuffle=False,num_workers=0)
val_dataset = MyDataSet(val_data['text'].values,characters=val_data['character'].values,
                        labels=val_data['labels'].values)
val_data_loader = DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=0)
test_dataset = MyDataSet(test['text'].values,characters=test['character'].values)
test_data_loader = DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=0)

def my_score(y_true, y_pred, normalize=True, sample_weight=None):
    error_squre = (y_pred - y_true)**2
    # sum_error_squre_root = np.sqrt(np.sum(error_squre))
    col_size = np.shape(y_pred)[1] if len(np.shape(y_true))>1 else 1
    total = np.shape(y_true)[0] * col_size

    error_norm = torch.sqrt(torch.sum(error_squre)/total)
    score = 1/(1+error_norm)
    return score

class MRModel(nn.Module):
    def __init__(self):
        super(MRModel, self).__init__()
        emb_dim = 768
        hid_dim = 128
        self.max_length = 128
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.electra_model = AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.fc1_0 = nn.Linear(emb_dim, hid_dim)  #第一层韵律标签的网络
        self.fc1_1 = nn.Linear(emb_dim, hid_dim) #第二层韵律标签的网络
        self.fc1_2 = nn.Linear(emb_dim, hid_dim) #第三层韵律标签的网络
        self.fc1_3 = nn.Linear(emb_dim, hid_dim) #第四层韵律标签的网络
        self.fc1_4 = nn.Linear(emb_dim, hid_dim) #第三层韵律标签的网络
        self.fc1_5 = nn.Linear(emb_dim, hid_dim) #第四层韵律标签的网络

        self.fc2_0 = nn.Linear(hid_dim, 1) #第一层韵律标签的网络
        self.fc2_1 = nn.Linear(hid_dim, 1) #第二层韵律标签的网络
        self.fc2_2 = nn.Linear(hid_dim, 1) #第三层韵律标签的网络
        self.fc2_3 = nn.Linear(hid_dim, 1) #第四层韵律标签的网络
        self.fc2_4 = nn.Linear(hid_dim, 1) #第三层韵律标签的网络
        self.fc2_5 = nn.Linear(hid_dim, 1) #第四层韵律标签的网络
        self.dropout = nn.Dropout(0.5)
        self.criterion = nn.MSELoss()
        # self.is_training = is_training  #True: training mode,False:inference mode

    def make_onehot(self,id,class_num):
        index = id.view(-1,1)
        one_hot = torch.zeros(index.size(0), class_num).to(index.device).scatter(dim=1,index=index,value=1)
        return one_hot

    def forward(self, texts,characters, label_id_layers=None,is_training=True):
        X_inputs = self.tokenizer(characters,texts, padding=True, return_tensors='pt', truncation=True, max_length=self.max_length)
        X_inputs.to(device)
        x = self.electra_model(X_inputs['input_ids'],
            token_type_ids=X_inputs['token_type_ids'],
            attention_mask=X_inputs['attention_mask'],
            return_dict=True)
        x = x.pooler_output
        # x = x[:,0,:]
        x = torch.reshape(x,[-1,x.size(-1)])

        fc1_out0 = F.relu(self.dropout(self.fc1_0(x)))
        out_layer_0 = self.fc2_0(fc1_out0) #第0个韵律层的输出

        fc1_out1 = F.relu(self.dropout(self.fc1_1(x)))  #第1个韵律层的第一层网络的输出
        out_layer_1 = self.fc2_1(fc1_out1)              #第1个韵律层的第二层网络的输出

        fc1_out2 = F.relu(self.dropout(self.fc1_2(x)))
        out_layer_2 = self.fc2_2(fc1_out2)  # 第2个韵律层的第二层网络的输出

        fc1_out3 = F.relu(self.dropout(self.fc1_3(x)))
        out_layer_3 = self.fc2_3(fc1_out3)  # 第3个韵律层的第二层网络的输出

        fc1_out4 = F.relu(self.dropout(self.fc1_4(x)))
        out_layer_4 = self.fc2_4(fc1_out4)  # 第3个韵律层的第二层网络的输出

        fc1_out5 = F.relu(self.dropout(self.fc1_5(x)))
        out_layer_5 = self.fc2_5(fc1_out5)  # 第3个韵律层的第二层网络的输出

        return out_layer_0,out_layer_1,out_layer_2,out_layer_3,out_layer_4,out_layer_5



def run_eval(mr_model,val_data_loader,has_label = True,is_train=True):
    mr_model.eval()
    texts_all,characters_all,labels_mat_merged, outputs_mat_merged = [],[], None, None
    for idx, data_batch in tqdm(enumerate(val_data_loader)):
        with torch.no_grad():
            texts = list(data_batch[0])
            texts_all.extend(texts)
            characters = data_batch[1]
            characters_all.extend(characters)
            if has_label:
                labels = torch.stack(data_batch[2], 0).float().to(device)
                labels_mat = labels.transpose(1, 0)
            outputs = mr_model(texts,characters)
            outputs = torch.stack(outputs, 0).squeeze()
            outputs_mat = outputs.transpose(1, 0)
            outputs_mat = outputs_mat.round()
            if idx == 0:
                if has_label:
                    labels_mat_merged = labels_mat
                outputs_mat_merged = outputs_mat

            else:
                if has_label:
                    labels_mat_merged = torch.cat([labels_mat_merged, labels_mat])
                outputs_mat_merged = torch.cat([outputs_mat_merged, outputs_mat])
    score = 0
    if has_label:
        score = my_score(labels_mat_merged, outputs_mat_merged)
        print('=================')
        print('val score={}'.format(score))
        if not is_train: #测试时生成对照结果文件
            with open('out_file.txt','w',encoding='utf-8') as wf:
                for i in range(len(texts_all)):
                    l = str(labels_mat_merged[i].cpu().numpy())
                    o = str(outputs_mat_merged[i].cpu().numpy())
                    right_flag = '' if l == o else 'FAIL'
                    wf.write(characters_all[i]+' '+texts_all[i]+' '+l+right_flag+'\n')
                    wf.write(characters_all[i]+' '+texts_all[i]+' '+o+right_flag+'\n')
    return score,outputs_mat_merged

def run_train(mr_model,train_data_loader):
    electa_params_ids = list(map(id, mr_model.electra_model.parameters()))
    base_params = filter(lambda p: id(p) not in electa_params_ids, mr_model.parameters())
    optimizer = optim.Adam([{'params': base_params},
                            {'params': mr_model.electra_model.parameters(), 'lr': 1e-05}],
                           lr=1e-04)
    best_epoch = 0
    best_val_score = 0
    for epoch in range(max_epoch):
        mr_model.train()
        for idx,data_batch in enumerate(tqdm(train_data_loader,desc='Epochs {}/{}'.format(epoch,max_epoch))):
            optimizer.zero_grad()
            texts = list(data_batch[0])
            characters = list(data_batch[1])
            labels = torch.stack(data_batch[2],0).float().to(device)
            # texts = torch.tensor(texts).to(device)
            # labels = labels.to(device)
            # input = trainer.X.to(device)
            # trainer.Y = torch.from_numpy(trainer.Y).to(trainer.device)
            outputs = mr_model(texts,characters)
            outputs = torch.stack(outputs,0).squeeze()
            loss_layer_0 = mr_model.criterion(outputs[0], labels[0])
            loss_layer_1 = mr_model.criterion(outputs[1], labels[1])
            loss_layer_2 = mr_model.criterion(outputs[2], labels[2])
            loss_layer_3 = mr_model.criterion(outputs[3], labels[3])
            loss_layer_4 = mr_model.criterion(outputs[4], labels[4])
            loss_layer_5 = mr_model.criterion(outputs[5], labels[5])
            loss = (loss_layer_0 + loss_layer_1 + loss_layer_2 + loss_layer_3 + loss_layer_4 + loss_layer_5) / 6
            loss.backward()
            optimizer.step()
            labels_mat = labels.transpose(1,0)
            outputs_mat = outputs.transpose(1,0)
            outputs_mat = outputs_mat.round()
            if idx == 0:
                labels_mat_merged = labels_mat
                outputs_mat_merged = outputs_mat
            else:
                labels_mat_merged = torch.cat([labels_mat_merged, labels_mat])
                outputs_mat_merged = torch.cat([outputs_mat_merged, outputs_mat])

            print('train loss={}'.format(loss))

        score = my_score(labels_mat_merged, outputs_mat_merged)
        print('+++++++++train score={} ++++++'.format(score))
        val_score,_ = run_eval(mr_model,val_data_loader)

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            print('best model of score {} at epoch {} saved'.format(val_score,epoch))
            torch.save(mr_model.state_dict(),model_path)
    print('best epoch=',best_epoch)
    print('best score=',best_val_score)

mr_model = MRModel()
mr_model.to(device)
if is_train:
    run_train(mr_model,train_data_loader)
else:
    if torch.cuda.is_available():
        model_state = torch.load(model_path)
    else:
        model_state = torch.load(model_path, map_location='cpu')
    mr_model.load_state_dict(model_state)
    val_score,val_pred = run_eval(mr_model,val_data_loader,is_train=is_train)
    print('val_score',val_score)
    _, predictions = run_eval(mr_model,test_data_loader,has_label=False)
    sub = submit.copy()
    # predictions_arr =  list(predictions.cpu().numpy())

    sub['emotion'] = list(predictions.cpu().numpy())
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(int(i)) for i in x]))
    sub.head()
    sub.to_csv('baseline_qa1.tsv', sep='\t', index=False)