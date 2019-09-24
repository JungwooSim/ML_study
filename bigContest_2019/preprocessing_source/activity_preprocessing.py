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
        if file.split('_')[1] == 'activity.csv':
            data_dict[file.split('.')[0]] = pd.read_csv(path_dir+file)

data_dict['train_activity']['kind'] = 'train'
data_dict['test1_activity']['kind'] = 'test1'
data_dict['test2_activity']['kind'] = 'test2'
data_dict['train_activity']['type_int'] = 1
data_dict['test1_activity']['type_int'] = 2
data_dict['test2_activity']['type_int'] = 3

activity_all = data_dict['train_activity'].append(data_dict['test1_activity'])
del(data_dict['train_activity'])
del(data_dict['test1_activity'])
activity_all = activity_all.append(data_dict['test2_activity'])
del(data_dict['test2_activity'])

# 유저 합계
activity_group = activity_all[[
    'acc_id',
    'playtime',
    'npc_kill',
    'solo_exp',
    'party_exp',
    'quest_exp',
    'rich_monster',
    'death',
    'revive',
    'exp_recovery',
    'fishing',
    'private_shop',
    'enchant_count'
]].groupby('acc_id').sum().reset_index()

# 유저 game_mone_change 절대값으로 변경 후 합계
activity_all['game_money_change_abs'] = abs(activity_all['game_money_change'])
activity_group = pd.merge(
    activity_group,
    activity_all[['acc_id','game_money_change_abs']].groupby('acc_id').sum().reset_index().rename(columns={'game_money_change_abs' : 'game_money_change'}),
    on="acc_id",
    how='left'
)

# 유저별 기록된 열 개수 (activity row count)
activity_group = pd.merge(
    activity_group,
    activity_all[['acc_id','day']].groupby('acc_id').count().reset_index().rename(columns={'day':'activity_row_count'}),
    on="acc_id",
    how='left'
)

# 유저 케릭터 수
activity_group = pd.merge(
    activity_group,
    activity_all.groupby('acc_id')['char_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'char_id' : 'charid_sum_detail'}),
    on="acc_id",
    how='left'
)
activity_group['charid_sum_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

activity_group['charid_sum_count'] = activity_group['charid_sum_detail'].apply(lambda x : len(set(x.split(','))))
activity_group['charid_sum_detail'] = activity_group['charid_sum_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저가 활동중인 서버와 서버 리스트
activity_group = pd.merge(
    activity_group,
    activity_all.groupby('acc_id')['server'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'server' : 'server_sum_detail'}),
    on="acc_id",
    how='left'
)
activity_group['server_sum_detail'].fillna('',inplace=True)

activity_group['server_sum_count'] = activity_group['server_sum_detail'].apply(lambda x : len(set(x.split(','))))
activity_group['server_sum_detail'] = activity_group['server_sum_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 기록된 날짜 개수와 날짜 리스트
activity_group = pd.merge(
    activity_group,
    activity_all.groupby('acc_id')['day'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'day' : 'day_sum_detail'}),
    on="acc_id",
    how='left'
)
activity_group['day_sum_detail'].fillna('',inplace=True)

activity_group['day_sum_count'] = activity_group['day_sum_detail'].apply(lambda x : len(set(x.split(','))))
activity_group['day_sum_detail'] = activity_group['day_sum_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# kind(train, test1, test2)
activity_group = pd.merge(
    activity_group,
    activity_all.groupby('acc_id')['kind'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'kind' : 'kind_sum_detail'}),
    on="acc_id",
    how='left'
)
activity_group['kind_sum_detail'].fillna('',inplace=True)

activity_group['kind_sum_count'] = activity_group['kind_sum_detail'].apply(lambda x : len(set(x.split(','))))
activity_group['kind_sum_detail'] = activity_group['kind_sum_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 열에서 0값을 가진 개수
activity_group['zero_cnt_activity'] = activity_group.apply(lambda x : x==0).sum(axis=1)

# 전처리 완료된 CSV 파일 생성
save_dir = os.getcwd()+"/arrangement_file/"
activity_group.to_csv(save_dir+"activity.csv", mode='w', index=False)