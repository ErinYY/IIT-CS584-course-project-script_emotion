import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import math
is_train = True #True: 训练模型，False:加载模型进行预测


from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs,ClassificationModel,ClassificationArgs
)

def my_score(y_true, y_pred, normalize=True, sample_weight=None):
    error_squre = (y_pred - y_true)**2
    # sum_error_squre_root = np.sqrt(np.sum(error_squre))
    col_size = np.shape(y_pred)[1] if len(np.shape(y_true))>1 else 1
    total = np.shape(y_true)[0] * col_size
    error_norm = np.sqrt(np.sum(error_squre)/total)
    score = 1/(1+error_norm)
    return score



class My_Model(ClassificationModel):
    def compute_metrics(
        self, preds, model_outputs, labels, eval_examples, multi_label=False, **kwargs
    ):
        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels,preds)
        return {**extra_metrics},["NA"]

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

train['text'] = train['content'].astype(str) + ' 角色: ' + train['character'].astype(str)
test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)




train['labels_temp'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])

full_data = train[['text', 'labels_temp']].copy()
print(full_data.head())
labels = []
# np.array(full_data['labels_temp'])[:,0]


full_data_shuff = full_data.sample(frac=1, random_state=42)
split_pos = int(len(full_data_shuff)*0.8)
train_data = full_data_shuff.iloc[:split_pos,:]
val_data = full_data_shuff.iloc[split_pos:,:]
print(train_data.head())
train_dfs = []
for i in range(6):
    train_df = train_data.copy()

    train_df['labels'] = train_df['labels_temp'].apply(lambda x:x[i])
    train_dfs.append(train_df)
    print(i,train_df['labels'].value_counts())
    # print(train_df)

val_dfs = []
for i in range(6):
    val_df = val_data.copy()
    val_df['labels'] = val_df['labels_temp'].apply(lambda x:x[i])
    val_dfs.append(val_df)
    print(val_df)
# model_args = MultiLabelClassificationArgs()
model_args = ClassificationArgs()
model_args.max_seq_length = 128
model_args.num_train_epochs = 1
model_args.no_save = False
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.overwrite_output_dir = True
model_args.regression = True
# model_args.use_multiprocessing = False
# model_args.process_count = 1




# model = MultiLabelClassificationModel('bert', model_path_or_name,args=model_args, num_labels=6) My_Model
# model = My_Model('bert', model_path_or_name,args=model_args, num_labels=6)
models = []
scores = []
pred_outputs_int = []
for i in range(6):
    if is_train:
        model_path_or_name = 'hfl/chinese-bert-wwm-ext'
    else:
        model_path_or_name = 'outputs'+str(i)
    model_args.cache_dir = 'cache_dir'+str(i)
    model_args.output_dir = 'outputs'+str(i)
    model = ClassificationModel('bert', model_path_or_name,num_labels=1,args=model_args)
    models.append(model)
    if is_train:
        model.train_model(train_dfs[i],acc=my_score)
    score, pred_outputs, wrong_preds = model.eval_model(val_dfs[i], score=my_score)
    print(score)
    round_vec = np.vectorize(round)
    pred_outputs_int.append(round_vec(pred_outputs))
    scores.append(score)
# pred_outputs = np.array(pred_outputs)
pred_outputs_int_merged = list(zip(pred_outputs_int[0],pred_outputs_int[1],pred_outputs_int[2],
                                   pred_outputs_int[3],pred_outputs_int[4],pred_outputs_int[5]))
# print('===============================')
# print(pred_outputs_int_merged)
val_labels = list(val_dfs[0]['labels_temp'].values)
val_labels = np.array(val_labels)
pred_outputs_int_merged = np.array(pred_outputs_int_merged)
final_score = my_score(val_labels,pred_outputs_int_merged)
print('final score:',final_score)
for i in range(6):
    print(scores[i])
# val_preds,val_raw_outputs = model.predict([text for text in val_data['text'].values])
# val_labels = list(val_data['labels'].values)
# val_preds = np.array(val_preds)
# val_labels = np.array(val_labels)
# val_errors = (val_preds-val_labels)**2
# val_rmse = np.sqrt(np.sum(val_errors)/(6*len(val_labels)))
# val_score = 1/(1+val_rmse)
# print('val_score={}'.format(val_score))
# predictions, raw_outputs = model.predict([text for text in test['text'].values])

# sub = submit.copy()
# sub['emotion'] = predictions
# sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
# sub.head()
#
# sub.to_csv('baseline.tsv', sep='\t', index=False)

