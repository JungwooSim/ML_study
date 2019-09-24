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
        if file.split('_')[1] == 'trade.csv':
            data_dict[file.split('.')[0]] = pd.read_csv(path_dir+file)

data_dict['train_trade']['kind'] = 'train'
data_dict['test1_trade']['kind'] = 'test1'
data_dict['test2_trade']['kind'] = 'test2'
data_dict['train_trade']['type_int'] = 1
data_dict['test1_trade']['type_int'] = 2
data_dict['test2_trade']['type_int'] = 3

trade_all = data_dict['train_trade'].append(data_dict['test1_trade'])
del(data_dict['train_trade'])
del(data_dict['test1_trade'])
trade_all = trade_all.append(data_dict['test2_trade'])
del(data_dict['test2_trade'])

# 틀이될 DF 생성
trade_group_train = pd.DataFrame(
    data = {
        "acc_id" : list(set(np.concatenate((trade_all.loc[trade_all.kind=='train']['source_acc_id'], trade_all.loc[trade_all.kind=='train']['target_acc_id']), axis=0).tolist())),
        'kind' : 'train'
    }
)

trade_group_test1 = pd.DataFrame(
    data = {
        "acc_id" : list(set(np.concatenate((trade_all.loc[trade_all.kind=='test1']['source_acc_id'], trade_all.loc[trade_all.kind=='test1']['target_acc_id']), axis=0).tolist())),
        'kind' : 'test1'
    }
)

trade_group_test2 = pd.DataFrame(
    data = {
        "acc_id" : list(set(np.concatenate((trade_all.loc[trade_all.kind=='test2']['source_acc_id'], trade_all.loc[trade_all.kind=='test2']['target_acc_id']), axis=0).tolist())),
        'kind' : 'test2'
    }
)
trade_group = pd.concat([trade_group_train, trade_group_test1, trade_group_test2], axis = 0)

# 기록된 날짜 개수와 날짜 리스트
trade_group = pd.merge(
    trade_group,
    trade_all.groupby(['source_acc_id','kind'])['day'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'source_acc_id' : 'acc_id', 'day' : 'day_sum_detail_trade'}),
    on=["acc_id",'kind'],
    how='left'
)
trade_group['day_sum_detail_trade'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

trade_group['day_sum_detail_trade'] = trade_group['day_sum_detail_trade'] + ',' + pd.merge(
    trade_group,
    trade_all.groupby(['target_acc_id','kind'])['day'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'target_acc_id' : 'acc_id', 'day' : 'day_sum_detail_trade'}),
    on=["acc_id",'kind'],
    how='left'
)['day_sum_detail_trade_y'].fillna('')

trade_group['day_sum_detail_trade'].replace(to_replace='^,|,$', value='', regex=True, inplace=True) #불필요한 콤마 제거

trade_group['day_sum_count_trade'] = trade_group['day_sum_detail_trade'].apply(lambda x : len(set(x.split(','))))
trade_group['day_sum_detail_trade'] = trade_group['day_sum_detail_trade'].apply(lambda x : ','.join(list(set(x.split(',')))))

trade_group['day_sum_count_trade'] = np.where(trade_group['day_sum_detail_trade'] == '',0,trade_group['day_sum_count_trade'])

# 유저별 서버와 서버 리스트
trade_group = pd.merge(
    trade_group,
    trade_all.groupby(['source_acc_id','kind'])['server'].apply(lambda group_series: ','.join(group_series.tolist())).reset_index().rename(columns={'source_acc_id' : 'acc_id', 'server' : 'server_sum_detail_trade'}),
    on=["acc_id",'kind'],
    how='left'
)
trade_group['server_sum_detail_trade'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

trade_group['server_sum_detail_trade'] = trade_group['server_sum_detail_trade'] + ',' + pd.merge(
    trade_group,
    trade_all.groupby(['target_acc_id','kind'])['server'].apply(lambda group_series: ','.join(group_series.tolist())).reset_index().rename(columns={'target_acc_id' : 'acc_id', 'server' : 'server_sum_detail_trade'}),
    on=["acc_id",'kind'],
    how='left'
)['server_sum_detail_trade_y'].fillna('')

trade_group['server_sum_detail_trade'].replace(to_replace='^,|,$', value='', regex=True, inplace=True) #불필요한 콤마 제거

trade_group['server_sum_count_trade'] = trade_group['server_sum_detail_trade'].apply(lambda x : len(set(x.split(','))))
trade_group['server_sum_detail_trade'] = trade_group['server_sum_detail_trade'].apply(lambda x : ','.join(list(set(x.split(',')))))

trade_group['server_sum_count_trade'] = np.where(trade_group['server_sum_detail_trade'] == '',0,trade_group['server_sum_count_trade'])

# 유저가 거래한 케릭터 개수와 수
trade_group = pd.merge(
    trade_group,
    trade_all.groupby(['source_acc_id','kind'])['source_char_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'source_acc_id' : 'acc_id', 'source_char_id' : 'char_sum_detail_trade'}),
    on=["acc_id",'kind'],
    how='left'
)
trade_group['char_sum_detail_trade'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

trade_group['char_sum_detail_trade'] = trade_group['char_sum_detail_trade'] + ',' + pd.merge(
    trade_group,
    trade_all.groupby(['target_acc_id','kind'])['target_char_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'target_acc_id' : 'acc_id', 'target_char_id' : 'char_sum_detail_trade'}),
    on=["acc_id",'kind'],
    how='left'
)['char_sum_detail_trade_y'].fillna('')

trade_group['char_sum_detail_trade'].replace(to_replace='^,|,$', value='', regex=True, inplace=True) #불필요한 콤마 제거

trade_group['char_sum_count_trade'] = trade_group['char_sum_detail_trade'].apply(lambda x : len(set(x.split(','))))
trade_group['char_sum_detail_trade'] = trade_group['char_sum_detail_trade'].apply(lambda x : ','.join(list(set(x.split(',')))))

trade_group['char_sum_count_trade'] = np.where(trade_group['char_sum_detail_trade'] == '',0,trade_group['char_sum_count_trade'])

# (교환+개인상점)을 했던 유저 수
trade_group = pd.merge(
    trade_group,
    trade_all[trade_all.type == 1].groupby(['source_acc_id','kind'])['target_acc_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'source_acc_id' : 'acc_id', 'target_acc_id' : 'trade_accid_detail'}),
    on=["acc_id",'kind'],
    how='left'
)
trade_group['trade_accid_detail'].fillna('',inplace=True) # 문자열을 + 연산으로 합칠 예정이므로 동일한 자료형으로 변경

trade_group['trade_accid_detail'] = trade_group['trade_accid_detail'] + ',' + pd.merge(
    trade_group,
    trade_all[trade_all.type == 1].groupby(['target_acc_id','kind'])['source_acc_id'].apply(lambda group_series: ','.join(group_series.astype(str).tolist())).reset_index().rename(columns={'target_acc_id' : 'acc_id', 'source_acc_id' : 'trade_accid_detail'}),
    on=["acc_id",'kind'],
    how='left'
)['trade_accid_detail_y'].fillna('')

trade_group['trade_accid_detail'].replace(to_replace='^,|,$', value='', regex=True, inplace=True) #불필요한 콤마 제거

trade_group['trade_accid_account'] = trade_group['trade_accid_detail'].apply(lambda x : len(set(x.split(','))))
trade_group['trade_accid_detail'] = trade_group['trade_accid_detail'].apply(lambda x : ','.join(list(set(x.split(',')))))

# 유저가 거래한 유저의 수
# 빈칸인 애들도 1로 기록되어서 빈칸 0으로 바꿔줄것이다
# 빈칸의 의미는 상점거래만 했다는것
trade_group['type_1_accid_account'] = np.where(trade_group['trade_accid_detail'] == '',0,trade_group['type_1_accid_account'])

# 교환창으로 주거나/판매한 횟수
trade_group = pd.merge(
    trade_group,
    trade_all[trade_all['type'] == 1][['source_acc_id','kind','type']].groupby(['source_acc_id', 'kind']).count().reset_index().rename(columns={'source_acc_id' : 'acc_id', 'type' : 'trade_source_count'}),
    on=["acc_id",'kind'],
    how='left'
).fillna(0)

# 교환창으로 받거나/구매한 횟수
trade_group = pd.merge(
    trade_group,
    trade_all[trade_all['type'] == 1][['target_acc_id','kind','type']].groupby(['target_acc_id','kind']).count().reset_index().rename(columns={'target_acc_id' : 'acc_id', 'type' : 'trade_target_count'}),
    on=["acc_id",'kind'],
    how='left'
).fillna(0)

# 교환창을 통해 주거나/판매한 횟수 + 받거나/구매한 횟수
trade_group['type_1'] = trade_group['trade_source_count'] + trade_group['trade_target_count']

# 상점 거래로 판매한 횟수
trade_group =pd.merge(
    trade_group,
    trade_all[trade_all['type'] == 0][['source_acc_id','kind','type']].groupby(['source_acc_id', 'kind']).count().reset_index().rename(columns={'source_acc_id' : 'acc_id', 'type' : 'type_0_source_count'}),
    on=["acc_id",'kind'],
    how='left'
).fillna(0)

# 상점 거래로 구매한 횟수
trade_group = pd.merge(
    trade_group,
    trade_all[trade_all['type'] == 0][['target_acc_id','kind','type']].groupby(['target_acc_id','kind']).count().reset_index().rename(columns={'target_acc_id' : 'acc_id', 'type' : 'type_0_target_count'}),
    on=["acc_id",'kind'],
    how='left'
).fillna(0)

# 상점을 통해 판매한 횟수 + 구매한 횟수
trade_group['type_0'] = trade_group['type_0_source_count'] + trade_group['type_0_target_count']

# 교환과 상점을 통해 거래한
trade_group['type_total_sum'] = trade_group['trade_source_count'] + trade_group['trade_target_count'] +trade_group['type_0_source_count'] + trade_group['type_0_target_count']

# 상점 오픈 유무
trade_group['type_0_is_open'] = np.where(trade_group['type_0_source_count'] == 0, 0, 1)

# 아이템을 받거나 구매한 양
trade_group = pd.merge(
    trade_group,
    trade_all.groupby(
        ['target_acc_id','kind','item_type']
    )['item_amount'].sum().reset_index().rename(
        columns={'target_acc_id':'acc_id'}
    ).pivot_table(
        index = ['acc_id','kind'],
        columns = 'item_type',
        values = 'item_amount'
    ).reset_index().fillna(0).rename(
        columns={
            'accessory': 'item_accessory_count',
            'adena': 'item_adena_count',
            'armor': 'item_armor_count',
            'enchant_scroll': 'item_enchant_scroll_count',
            'etc': 'item_etc_count',
            'spell': 'item_spell_count',
            'weapon': 'item_weapon_count'
        }
    ),
    on=["acc_id",'kind'],
    how='left'
).fillna(0)

# 아이템을 팔거나 준것
trade_group = pd.merge(
    trade_group,
    trade_all.groupby(
        ['source_acc_id', 'kind', 'item_type']
    )['item_amount'].sum().reset_index().rename(
        columns={'source_acc_id': 'acc_id'}
    ).pivot_table(
        index=['acc_id', 'kind'],
        columns='item_type',
        values='item_amount'
    ).reset_index().fillna(0).rename(
        columns={
            'accessory': 'item_accessory_out_count',
            'adena': 'item_adena_out_count',
            'armor': 'item_armor_out_count',
            'enchant_scroll': 'item_enchant_scroll_out_count',
            'etc': 'item_etc_out_count',
            'spell': 'item_spell_out_count',
            'weapon': 'item_weapon_out_count'
        }
    ),
    on=["acc_id",'kind'],
    how='left'
).fillna(0)

# 상점 수입
trade_group = pd.merge(
    trade_group,
    trade_all[trade_all.type == 0].groupby(['source_acc_id','kind'])['item_price'].sum().reset_index().rename(columns={'source_acc_id':'acc_id', 'item_price': 'type_0_avenue'}),
    on=["acc_id",'kind'],
    how='left'
).fillna(0)

# 거래 유무
trade_group['not_any_trade'] = np.where((trade_group==0).sum(axis = 1) == (trade_group==0).sum(axis = 1).max(), 1, 0)

# 전처리 완료된 CSV 파일 생성
save_dir = os.getcwd()+"/arrangement_file/"
trade_group.to_csv(save_dir+"trade.csv", mode='w', index=False)