import numpy as np
import pandas as pd
from collections import OrderedDict
import os

os.chdir('..')
path_dir = os.getcwd()+"/champions/"
file_list = sorted(os.listdir(path_dir))

data_dict = OrderedDict()
for file in file_list:
    if file.find('csv') is not -1:
        if file.split('_')[1] == 'pledge.csv':
            data_dict[file.split('.')[0]] = pd.read_csv(path_dir+file)

data_dict['train_pledge']['type'] = 'train'
data_dict['test1_pledge']['type'] = 'test1'
data_dict['test2_pledge']['type'] = 'test2'
data_dict['train_pledge']['type_int'] = 1
data_dict['test1_pledge']['type_int'] = 2
data_dict['test2_pledge']['type_int'] = 3

pledge_all = data_dict['train_pledge'].append(data_dict['test1_pledge'])
del(data_dict['train_pledge'])
del(data_dict['test1_pledge'])
pledge_all = pledge_all.append(data_dict['test2_pledge'])
del(data_dict['test2_pledge'])

# 유저별 합계
group_sum_list = [
    'acc_id',
    'play_char_cnt',
    'combat_char_cnt',
    'pledge_combat_cnt',
    'random_attacker_cnt',
    'random_defender_cnt',
    'same_pledge_cnt',
    'temp_cnt',
    'etc_cnt',
    'combat_play_time',
    'non_combat_play_time'
]
pledge_group = pledge_all[group_sum_list].groupby('acc_id').sum().reset_index()

# 유저별 기록된 날짜 개수와 날짜 리스트
pledge_group = pd.merge(
    pledge_group,
    pledge_all.groupby('acc_id')['day'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'day' : 'day_concat_detail'}),
    on="acc_id",
    how='left'
)
pledge_group['day_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

pledge_group['day_concat_count'] = pledge_group['day_concat_detail'].apply(lambda x : len(set(x.split(','))))
pledge_group['day_concat_detail'] = pledge_group['day_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 케릭터와 케릭터 수
pledge_group = pd.merge(
    pledge_group,
    pledge_all.groupby('acc_id')['char_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'char_id' : 'charid_concat_detail'}),
    on="acc_id",
    how='left'
)
pledge_group['charid_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

pledge_group['charid_concat_count'] = pledge_group['charid_concat_detail'].apply(lambda x : len(set(x.split(','))))
pledge_group['charid_concat_detail'] = pledge_group['charid_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 서버와 서버 리스트
pledge_group = pd.merge(
    pledge_group,
    pledge_all.groupby('acc_id')['server'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'server' : 'server_concat_detail'}),
    on="acc_id",
    how='left'
)
pledge_group['server_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

pledge_group['server_concat_count'] = pledge_group['server_concat_detail'].apply(lambda x : len(set(x.split(','))))
pledge_group['server_concat_detail'] = pledge_group['server_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저가 가입한 혈맹과, 혈맹 수
pledge_group = pd.merge(
    pledge_group,
    pledge_all.groupby('acc_id')['pledge_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'pledge_id' : 'pledgeid_concat_detail'}),
    on="acc_id",
    how='left'
)
pledge_group['pledgeid_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

pledge_group['pledgeid_concat_count'] = pledge_group['pledgeid_concat_detail'].apply(lambda x : len(set(x.split(','))))
pledge_group['pledgeid_concat_detail'] = pledge_group['pledgeid_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# kind(train, test1, test2)
pledge_group = pd.merge(
    pledge_group,
    pledge_all.groupby('acc_id')['type'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'type' : 'kind_concat_detail'}),
    on="acc_id",
    how='left'
)
pledge_group['kind_concat_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

pledge_group['kind_concat_count'] = pledge_group['kind_concat_detail'].apply(lambda x : len(set(x.split(','))))
pledge_group['kind_concat_detail'] = pledge_group['kind_concat_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저별 기록된 열 개수
pledge_group = pd.merge(
    pledge_group,
    pledge_all[['acc_id','day']].groupby('acc_id').count().reset_index().rename(columns={'day':'pledge_row_count'}),
    on="acc_id",
    how='left'
)

# 게임에 접속한 혈맹원 수 평균
pledge_group = pd.merge(
    pledge_group,
    pledge_all[['acc_id', 'play_char_cnt']].groupby('acc_id').mean().reset_index().rename(columns={'play_char_cnt':'play_char_mean'}),
    on="acc_id",
    how='left'
)

# 전투에 참여한 혈매원 수 평균
pledge_group = pd.merge(
    pledge_group,
    pledge_all[['acc_id', 'combat_char_cnt']].groupby('acc_id').mean().reset_index().rename(columns={'combat_char_cnt':'combat_char_mean'}),
    on="acc_id",
    how='left'
)

# 게임에 접속한 혈맹원 수 평균 / 전투에 참여한 혈매원 수 평균 <- 전투에 참여하는 비율
pledge_all['play_combat_char_ratio'] = pledge_all['combat_char_cnt'] / pledge_all['play_char_cnt']
pledge_group = pd.merge(
    pledge_group,
    pledge_all[pledge_all['play_combat_char_ratio'] !=0][['acc_id','play_combat_char_ratio']].groupby('acc_id').mean().reset_index(),
    on="acc_id",
    how='left'
).fillna(0)

# 열에서 0값을 가진 개수
pledge_group['zero_cnt_pledge'] = pledge_group.apply(lambda x : x==0).sum(axis=1)

# 전처리 완료된 CSV 파일 생성
save_dir = os.getcwd()+"/arrangement_file/"
pledge_group.to_csv(save_dir+"pledge.csv", mode='w', index=False)