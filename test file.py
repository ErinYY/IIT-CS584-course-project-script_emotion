import pandas as pd


def padding(x):
    splitted = x.split('_')
    splitted[-1] = '{:0>4d}'.format(int(splitted[-1]))
    x = '_'.join(splitted)
    return x
def proc(dataset_filename,dataset_cid_filename,uniq_content_filename):

    pre_content=''
    train = pd.read_csv(dataset_filename, sep='\t')
    train['sort_id'] = train['id'].apply(padding)
    train = train.sort_values(by='sort_id')

    uniq_content=[]
    pre_id_head = ''
    train.insert(1,'content_id',None)
    for i in range(len(train)):
        content_i = train.iloc[i]['content']
        id_items = train.iloc[i]['id'].split('_')
        id_head = '_'.join(id_items[0:2])
        if id_head != pre_id_head: #剧本段落不同
            uniq_content.append('')
            uniq_content.append('')
            pre_id_head = id_head
        if content_i != pre_content:
            uniq_content.append(content_i)
            pre_content = content_i
        train.iloc[i,train.columns.get_loc('content_id')] = len(uniq_content)-1

    with open(uniq_content_filename,'w',encoding='utf-8') as wf:
        wf.writelines([line+'\n' for line in uniq_content])
    train = train.sort_index()
    train.to_csv(dataset_cid_filename,index_label=False,index=False)

if __name__=='__main__':
    #处理训练数据
    dataset_filename, dataset_cid_filename, uniq_content_filename=\
        'train_dataset_v2.tsv','train_dataset_with_contentid.tsv','train_uniq_content.tsv'
    proc(dataset_filename, dataset_cid_filename, uniq_content_filename)
    #处理测试数据
    dataset_filename, dataset_cid_filename, uniq_content_filename=\
        'test_dataset.tsv','test_dataset_with_contentid.tsv','test_uniq_content.tsv'
    proc(dataset_filename, dataset_cid_filename, uniq_content_filename)


