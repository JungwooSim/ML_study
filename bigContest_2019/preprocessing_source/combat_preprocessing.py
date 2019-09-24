import numpy as np
import pandas as pd
from collections import OrderedDict
import os
from statistics import mean

os.chdir('..')
path_dir = os.getcwd()+"/champions/"
file_list = sorted(os.listdir(path_dir))

data_dict = OrderedDict()
for file in file_list:
    if file.find('csv') is not -1:
        if file.split('_')[1] == 'combat.csv':
            data_dict[file.split('.')[0]] = pd.read_csv(path_dir+file)

data_dict['train_combat']['type'] = 'train'
data_dict['test1_combat']['type'] = 'test1'
data_dict['test2_combat']['type'] = 'test2'
data_dict['train_combat']['type_int'] = 1
data_dict['test1_combat']['type_int'] = 2
data_dict['test2_combat']['type_int'] = 3

combat_all = data_dict['train_combat'].append(data_dict['test1_combat'])
del(data_dict['train_combat'])
del(data_dict['test1_combat'])
combat_all = combat_all.append(data_dict['test2_combat'])
del(data_dict['test2_combat'])

# 유저별 합계
group_sum_list = [
    'acc_id',
    'day',
    'pledge_cnt',
    'random_attacker_cnt',
    'random_defender_cnt',
    'temp_cnt',
    'same_pledge_cnt',
    'etc_cnt',
    'num_opponent'
]
combat_group = combat_all[group_sum_list].groupby('acc_id').sum().reset_index()

# 유저별 케릭터 클래스와 클래스 수
combat_group = pd.merge(
    combat_group,
    combat_all.groupby('acc_id')['class'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'class' : 'class_concat_detail'}),
    on="acc_id",
    how='left'
)
combat_group['class_concat_detail'].fillna('',inplace=True)

combat_group['class_concat_count'] = combat_group['class_concat_detail'].apply(lambda x : len(set(x.split(','))))
combat_group['class_concat_detail'] = combat_group['class_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 케릭터와 케릭터 수
combat_group = pd.merge(
    combat_group,
    combat_all.groupby('acc_id')['char_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'char_id' : 'charid_concat_detail'}),
    on="acc_id",
    how='left'
)
combat_group['charid_concat_detail'].fillna('',inplace=True)

combat_group['charid_concat_count'] = combat_group['charid_concat_detail'].apply(lambda x : len(set(x.split(','))))
combat_group['charid_concat_detail'] = combat_group['charid_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 서버와 서버 리스트
combat_group = pd.merge(
    combat_group,
    combat_all.groupby('acc_id')['server'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'server' : 'server_concat_detail'}),
    on="acc_id",
    how='left'
)
combat_group['server_concat_detail'].fillna('',inplace=True)

combat_group['server_concat_count'] = combat_group['server_concat_detail'].apply(lambda x : len(set(x.split(','))))
combat_group['server_concat_detail'] = combat_group['server_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 케릭터 레벨과 레벨 리스트
combat_group = pd.merge(
    combat_group,
    combat_all.groupby('acc_id')['level'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'level' : 'level_concat_detail'}),
    on="acc_id",
    how='left'
)
combat_group['level_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

combat_group['level_concat_count'] = combat_group['level_concat_detail'].apply(lambda x : len(set(x.split(','))))
combat_group['level_concat_detail'] = combat_group['level_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 기록된 날짜 개수와 날짜 리스트
combat_group = pd.merge(
    combat_group,
    combat_all.groupby('acc_id')['day'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'day' : 'day_concat_detail'}),
    on="acc_id",
    how='left'
)
combat_group['day_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

combat_group['day_concat_count'] = combat_group['day_concat_detail'].apply(lambda x : len(set(x.split(','))))
combat_group['day_concat_detail'] = combat_group['day_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# kind(train, test1, test2)
combat_group = pd.merge(
    combat_group,
    combat_all.groupby('acc_id')['type'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'type' : 'kind_concat_detail'}),
    on="acc_id",
    how='left'
)
combat_group['kind_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

combat_group['kind_concat_count'] = combat_group['kind_concat_detail'].apply(lambda x : len(set(x.split(','))))
combat_group['kind_concat_detail'] = combat_group['kind_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저가 이용중인 케릭터 평균 레벨
combat_group['level_mean'] = combat_group['level_concat_detail'].apply(lambda x : mean(map(int, x.split(','))))

# 유저가 이용중인 케릭터 최고 레벨
combat_group['level_max'] = combat_group['level_concat_detail'].apply(lambda x : max(map(int, x.split(','))))

# 유저별 기록된 열 개수 (combat row count)
combat_group = pd.merge(
    combat_group,
    combat_all[['acc_id','day']].groupby('acc_id').count().reset_index().rename(columns={'day':'combat_row_count'}),
    on="acc_id",
    how='left'
)

# 열에서 0값을 가진 개수
combat_group['zero_cnt_combat'] = combat_group.apply(lambda x : x==0).sum(axis=1)

# 전처리 완료된 CSV 파일 생성
save_dir = os.getcwd()+"/arrangement_file/"
combat_group.to_csv(save_dir+"combat.csv", mode='w', index=False)