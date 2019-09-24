import numpy as np
import pandas as pd
from collections import OrderedDict
import os

os.chdir('..') #상위디렉터리로 이동
path_dir = os.getcwd()+"/champions/"
file_list = sorted(os.listdir(path_dir))

data_dict = OrderedDict()
for file in file_list:
    if file.find('csv') is not -1:
        if file.split('_')[1] == 'payment.csv':
            data_dict[file.split('.')[0]] = pd.read_csv(path_dir+file)

data_dict['train_payment']['type'] = 'train'
data_dict['test1_payment']['type'] = 'test1'
data_dict['test2_payment']['type'] = 'test2'

payment_all = data_dict['train_payment'].append(data_dict['test1_payment'])
del(data_dict['train_payment'])
del(data_dict['test1_payment'])
payment_all = payment_all.append(data_dict['test2_payment'])
del(data_dict['test2_payment'])

payment_group = payment_all.groupby('acc_id').sum().reset_index()

# 유저별 기록된 날짜 개수와 날짜 리스트
payment_group = pd.merge(
    payment_group,
    payment_all.groupby('acc_id')['day'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'day' : 'day_concat_detail'}),
    on="acc_id",
    how='left'
)
payment_group['day_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

payment_group['day_concat_count'] = payment_group['day_concat_detail'].apply(lambda x : len(set(x.split(','))))
payment_group['day_concat_detail'] = payment_group['day_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 기록된 열 개수
payment_group = pd.merge(
    payment_group,
    payment_all[['acc_id','day']].groupby('acc_id').count().reset_index().rename(columns={'day':'payment_row_count'}),
    on="acc_id",
    how='left'
)

# kind(train, test1, test2)
payment_group = pd.merge(
    payment_group,
    payment_all.groupby('acc_id')['type'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'type' : 'kind_concat_detail'}),
    on="acc_id",
    how='left'
)
payment_group['kind_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

payment_group['kind_concat_count'] = payment_group['kind_concat_detail'].apply(lambda x : len(set(x.split(','))))
payment_group['kind_concat_detail'] = payment_group['kind_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 평균결제금액
payment_group['amount_spent_mean'] = payment_group['amount_spent'] / payment_group['payment_row_count']

# 전처리 완료된 CSV 파일 생성
save_dir = os.getcwd()+"/arrangement_file/"
payment_group.to_csv(save_dir+"payment.csv", mode='w', index=False)