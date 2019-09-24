import numpy as np
import pandas as pd
from collections import OrderedDict
import os

train_label = pd.read_csv(os.getcwd()+"/champions/train_label.csv")

file_list = sorted(os.listdir(os.getcwd()+'/arrangement_file/'))
data_dict = OrderedDict()
for file in file_list:
    if file.find('csv') is not -1:
        if file.split('.')[0] in ['activity','combat','payment','pledge','trade'] :
            data_dict[file.split('.')[0]] = pd.read_csv(os.getcwd()+'/arrangement_file/'+file)

# 변수명 수정
data_dict['activity'].rename(
    columns={
        'day':"day_activity"
    }
)
data_dict['combat'].rename(
    columns={
        'day':'day_combat',
        'random_attacker_cnt':'random_attacker_cnt_combat',
        'random_defender_cnt':'random_defender_cnt_combat',
        'temp_cnt':'temp_cnt_combat',
        'same_pledge_cnt':'same_pledge_cnt_combat',
        'etc_cnt':'etc_cnt_combat',
        'charid_concat_detail':'charid_concat_detail_combat',
        'charid_concat_count':'charid_concat_count_combat',
        'server_concat_detail':'server_concat_detail_combat',
        'server_concat_count':'server_concat_count_combat',
        'day_concat_detail':'day_concat_detail_combat',
        'day_concat_count':'day_concat_count_combat',
        'kind_concat_detail':'kind_concat_detail_combat',
        'kind_concat_count':'kind_concat_count_combat'
    },
    inplace=True
)

data_dict['payment'].rename(
    columns={
        'day':'day_payment',
        'day_concat_detail':'day_concat_detail_payment',
        'day_concat_count':'day_concat_count_payment',
        'kind_concat_detail':'kind_concat_detail_payment',
        'kind_concat_count':'kind_concat_count_payment'
    },
    inplace=True
)
data_dict['pledge'].rename(
    columns={
        'random_attacker_cnt':'random_attacker_cnt_pledge',
        'random_defender_cnt':'random_defender_cnt_pledge',
        'same_pledge_cnt':'same_pledge_cnt_pledge',
        'temp_cnt':'temp_cnt_pledge',
        'etc_cnt':'etc_cnt_pledge',
        'combat_play_time':'combat_play_time_pledge',
        'non_combat_play_time':'non_combat_play_time_pledge',
        'day_concat_detail':'day_concat_detail_pledge',
        'day_concat_count':'day_concat_count_pledge',
        'charid_concat_detail':'charid_concat_detail_pledge',
        'charid_concat_count':'charid_concat_count_pledge',
        'server_concat_detail':'server_concat_detail_pledge',
        'server_concat_count':'server_concat_count_pledge',
        'pledgeid_concat_detail':'pledgeid_concat_detail_pledge',
        'pledgeid_concat_count':'pledgeid_concat_count_pledge',
        'kind_concat_detail':'kind_concat_detail_pledge',
        'kind_concat_count':'kind_concat_count_pledge_pledge'
    },
    inplace=True
)

# train
train = pd.merge(
    train_label.rename(columns={"survival_time":"target_survival_time","amount_spent":"target_amount_spent"}),
    data_dict['activity'],
    on="acc_id",
    how='left'
)

train = pd.merge(
    train,
    data_dict['combat'],
    on="acc_id",
    how='left',
    suffixes=('_activity', '_combat')
)

train = pd.merge(
    train,
    data_dict['payment'],
    on="acc_id",
    how='left',
    suffixes=('_combat', '_payment')
)

train = pd.merge(
    train,
    data_dict['pledge'],
    on="acc_id",
    how='left',
    suffixes=('_payment', '_pledge')
)
train = pd.merge(
    train,
    data_dict['trade'][data_dict['trade']['kind']=='train'],
    on="acc_id",
    how='left',
    suffixes=('_pledge', '_trade')
)

# test1
test1 = data_dict['activity'][data_dict['activity']['kind_sum_detail'] == 'test1']
test1 = pd.merge(
    test1,
    data_dict['combat'],
    on="acc_id",
    how='left',
    suffixes=('_activity', '_combat')
)
test1 = pd.merge(
    test1,
    data_dict['payment'],
    on="acc_id",
    how='left',
    suffixes=('_combat', '_payment')
)

test1 = pd.merge(
    test1,
    data_dict['pledge'],
    on="acc_id",
    how='left',
    suffixes=('_payment', '_pledge')
)
test1 = pd.merge(
    test1,
    data_dict['trade'][data_dict['trade']['kind']=='test1'],
    on="acc_id",
    how='left',
    suffixes=('_pledge', '_trade')
)

# test2
test2 = data_dict['activity'][data_dict['activity']['kind_sum_detail'] == 'test2']
test2 = pd.merge(
    test2,
    data_dict['combat'],
    on="acc_id",
    how='left',
    suffixes=('_activity', '_combat')
)
test2 = pd.merge(
    test2,
    data_dict['payment'],
    on="acc_id",
    how='left',
    suffixes=('_combat', '_payment')
)
test2 = pd.merge(
    test2,
    data_dict['pledge'],
    on="acc_id",
    how='left',
    suffixes=('_payment', '_pledge')
)
test2 = pd.merge(
    test2,
    data_dict['trade'][data_dict['trade']['kind']=='test2'],
    on="acc_id",
    how='left',
    suffixes=('_pledge', '_trade')
)

# nan fill
def nan_fill(dataframe):
    # zero_cnt_pledge는 16
    dataframe['zero_cnt_pledge'] = dataframe['zero_cnt_pledge'].fillna(16)
    #나머지는 다 0로 대체
    dataframe = dataframe.fillna(0)
    
    return dataframe

train = nan_fill(train)
test1 = nan_fill(test1)
test2 = nan_fill(test2)

# 결제 여부
train['not_any_payment'] = np.where(train['payment_row_count'] == 0, 1, 0)
test1['not_any_payment'] = np.where(test1['payment_row_count'] == 0, 1, 0)
test2['not_any_payment'] = np.where(test2['payment_row_count'] == 0, 1, 0)

# 혈맹 활동 여부
train['non_any_pledge'] = np.where(train['zero_cnt_pledge'] == 16,1, 0)
test1['non_any_pledge'] = np.where(test1['zero_cnt_pledge'] == 16,1, 0)
test2['non_any_pledge'] = np.where(test2['zero_cnt_pledge'] == 16,1, 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 구입한 아이템들 총 합
in_item_sum_list = ['item_accessory_count','item_adena_count', 'item_armor_count', 'item_enchant_scroll_count','item_etc_count','item_spell_count','item_weapon_count']
train['in_item_sum'] = scaler.fit_transform(np.log1p(train[in_item_sum_list])).sum(axis = 1)
test1['in_item_sum'] = scaler.fit_transform(np.log1p(test1[in_item_sum_list])).sum(axis = 1)
test2['in_item_sum'] = scaler.fit_transform(np.log1p(test2[in_item_sum_list])).sum(axis = 1)

# 판매한 아이템들 총 합
out_item_sum_list = ['item_accessory_out_count','item_adena_out_count','item_armor_out_count','item_enchant_scroll_out_count','item_etc_out_count','item_spell_out_count','item_weapon_out_count']
train['out_item_sum'] = scaler.fit_transform(np.log1p(train[out_item_sum_list])).sum(axis = 1)
test1['out_item_sum'] = scaler.fit_transform(np.log1p(test1[out_item_sum_list])).sum(axis = 1)
test2['out_item_sum'] = scaler.fit_transform(np.log1p(test2[out_item_sum_list])).sum(axis = 1)

# 활동 내역 총 합
activit_list = ['playtime','npc_kill','solo_exp','party_exp','quest_exp','rich_monster','death','revive','exp_recovery','fishing','private_shop','enchant_count','game_money_change','activity_row_count','charid_sum_count','server_sum_count','day_sum_count'] #except 'zero_cnt_activity'제외
train['activity_summarise'] = scaler.fit_transform(np.log1p(train[activit_list])).sum(axis=1)
test1['activity_summarise'] = scaler.fit_transform(np.log1p(test1[activit_list])).sum(axis=1)
test2['activity_summarise'] = scaler.fit_transform(np.log1p(test2[activit_list])).sum(axis=1)

# 전투내역 총합
combat_list = ['pledge_cnt','random_attacker_cnt_combat','random_defender_cnt_combat','temp_cnt_combat','same_pledge_cnt_combat','etc_cnt_combat','num_opponent','class_concat_count','charid_concat_count_combat','server_concat_count_combat','level_concat_count','day_concat_count_combat','level_mean','level_max','combat_row_count'] #except 'day_combat', 'zero_cnt_combat'
train['combat_summarise'] = scaler.fit_transform(np.log1p(train[combat_list])).sum(axis=1)
test1['combat_summarise'] = scaler.fit_transform(np.log1p(test1[combat_list])).sum(axis=1)
test2['combat_summarise'] = scaler.fit_transform(np.log1p(test2[combat_list])).sum(axis=1)

# 혈맹 활동 내역 총합
pledge_list = ['play_char_cnt', 'combat_char_cnt', 'pledge_combat_cnt', 'random_attacker_cnt_pledge', 'random_defender_cnt_pledge','same_pledge_cnt_pledge','temp_cnt_pledge', 'etc_cnt_pledge', 'combat_play_time_pledge', 'non_combat_play_time_pledge','day_concat_count_pledge','charid_concat_count_pledge','server_concat_count_pledge', 'pledgeid_concat_count_pledge','pledge_row_count', 'play_char_mean','combat_char_mean', 'play_combat_char_ratio']# except'zero_cnt_pledge'
train['pledge_summarise'] = scaler.fit_transform(np.log1p(train[pledge_list])).sum(axis=1)
test1['pledge_summarise'] = scaler.fit_transform(np.log1p(test1[pledge_list])).sum(axis=1)
test2['pledge_summarise'] = scaler.fit_transform(np.log1p(test2[pledge_list])).sum(axis=1)

# 활동 내역중 0 이 아닌 개수 합
train['activity_active_count'] = (train[activit_list]>0).sum(axis=1)
test1['activity_active_count'] = (test1[activit_list]>0).sum(axis=1)
test2['activity_active_count'] = (test2[activit_list]>0).sum(axis=1)

# 전투 내역중 0 이 아닌 개수 합
train['combat_active_count'] = (train[combat_list]>0).sum(axis=1)
test1['combat_active_count'] = (test1[combat_list]>0).sum(axis=1)
test2['combat_active_count'] = (test2[combat_list]>0).sum(axis=1)

# 혈맹 활동 내역중 0 이 아닌 개수 합
train['pledge_active_count'] = (train[pledge_list]>0).sum(axis=1)
test1['pledge_active_count'] = (test1[pledge_list]>0).sum(axis=1)
test2['pledge_active_count'] = (test2[pledge_list]>0).sum(axis=1)

# 하루 평균 플레이 시간 - 플레이 시간 / 총 플레이 기간
train['playtime_day'] = train['playtime'] / train['day_sum_count']
test1['playtime_day'] = test1['playtime'] / test1['day_sum_count']
test2['playtime_day'] = test2['playtime'] / test2['day_sum_count']

# 하루 평균 결제 금액
train['amount_spent_day'] = train['amount_spent'] / train['day_sum_count']
test1['amount_spent_day'] = test1['amount_spent'] / test1['day_sum_count']
test2['amount_spent_day'] = test2['amount_spent'] / test2['day_sum_count']

## 각 주별 충성도 - 한 주 동안의 접속일(중복제거) / 7
train['royalty_week_1'] = train['royalty_week_2'] = train['royalty_week_3'] = train['royalty_week_4'] = 0
for i in range(1,29) :
    if i >= 22 :
        train['royalty_week_4'] = train['royalty_week_4'] + train['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    elif i >= 15 :
        train['royalty_week_3'] = train['royalty_week_3'] + train['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    elif i >= 8 :
        train['royalty_week_2'] = train['royalty_week_2'] + train['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    else :
        train['royalty_week_1'] = train['royalty_week_1'] + train['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)

train['royalty_week_4'] = train['royalty_week_4'] / 7
train['royalty_week_3'] = train['royalty_week_3'] / 7
train['royalty_week_2'] = train['royalty_week_2'] / 7
train['royalty_week_1'] = train['royalty_week_1'] / 7

## 각 주별 충성도 - 한 주 동안의 접속일(중복제거) / 7
test1['royalty_week_1'] = test1['royalty_week_2'] = test1['royalty_week_3'] = test1['royalty_week_4'] = 0
for i in range(1,29) :
    if i >= 22 :
        test1['royalty_week_4'] = test1['royalty_week_4'] + test1['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    elif i >= 15 :
        test1['royalty_week_3'] = test1['royalty_week_3'] + test1['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    elif i >= 8 :
        test1['royalty_week_2'] = test1['royalty_week_2'] + test1['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    else :
        test1['royalty_week_1'] = test1['royalty_week_1'] + test1['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)

test1['royalty_week_4'] = test1['royalty_week_4'] / 7
test1['royalty_week_3'] = test1['royalty_week_3'] / 7
test1['royalty_week_2'] = test1['royalty_week_2'] / 7
test1['royalty_week_1'] = test1['royalty_week_1'] / 7

test2['royalty_week_1'] = test2['royalty_week_2'] = test2['royalty_week_3'] = test2['royalty_week_4'] = 0
for i in range(1,29) :
    if i >= 22 :
        test2['royalty_week_4'] = test2['royalty_week_4'] + test2['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    elif i >= 15 :
        test2['royalty_week_3'] = test2['royalty_week_3'] + test2['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    elif i >= 8 :
        test2['royalty_week_2'] = test2['royalty_week_2'] + test2['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)
    else :
        test2['royalty_week_1'] = test2['royalty_week_1'] + test2['day_sum_detail'].apply(lambda x : 1 if str(i) in x.split(',') else 0)

test2['royalty_week_4'] = test2['royalty_week_4'] / 7
test2['royalty_week_3'] = test2['royalty_week_3'] / 7
test2['royalty_week_2'] = test2['royalty_week_2'] / 7
test2['royalty_week_1'] = test2['royalty_week_1'] / 7

# 분석 과정에서 얻은 정보를 통해 survival 체크하는 변수 생성
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(
    train[[
        'playtime',
        'private_shop',
        'activity_row_count',
        'day_sum_count',
        'day_combat',
        'day_concat_count_combat',
        'combat_row_count',
        'day_sum_count_trade',
    ]]
)
X_minMax_scaled = minMaxScaler.transform(
    train[[
        'playtime',
        'private_shop',
        'activity_row_count',
        'day_sum_count',
        'day_combat',
        'day_concat_count_combat',
        'combat_row_count',
        'day_sum_count_trade',
    ]]
)
train['survival_check_val_1'] = pd.DataFrame(X_minMax_scaled).sum(axis=1).tolist()

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(
    test1[[
        'playtime',
        'private_shop',
        'activity_row_count',
        'day_sum_count',
        'day_combat',
        'day_concat_count_combat',
        'combat_row_count',
        'day_sum_count_trade',
    ]]
)
X_minMax_scaled = minMaxScaler.transform(
    test1[[
        'playtime',
        'private_shop',
        'activity_row_count',
        'day_sum_count',
        'day_combat',
        'day_concat_count_combat',
        'combat_row_count',
        'day_sum_count_trade',
    ]]
)
test1['survival_check_val_1'] = pd.DataFrame(X_minMax_scaled).sum(axis=1).tolist()

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(
    test2[[
        'playtime',
        'private_shop',
        'activity_row_count',
        'day_sum_count',
        'day_combat',
        'day_concat_count_combat',
        'combat_row_count',
        'day_sum_count_trade',
    ]]
)
X_minMax_scaled = minMaxScaler.transform(
    test2[[
        'playtime',
        'private_shop',
        'activity_row_count',
        'day_sum_count',
        'day_combat',
        'day_concat_count_combat',
        'combat_row_count',
        'day_sum_count_trade',
    ]]
)
test2['survival_check_val_1'] = pd.DataFrame(X_minMax_scaled).sum(axis=1).tolist()

# 마지막 주에 일별로 접속 여부 변수 생성(게임을 그만두는 사람은 점차적으로 그만 뒀을거라 생각)
day_sum = train['day_sum_detail']
day_sum = day_sum.apply(lambda x: x.split(","))
for i in range (len(day_sum)):
    day_sum[i] = list(map(int, day_sum[i]))
train['detail'] = day_sum
for i in range (1, 29):
    train['day_'+str(i)] = train['detail'].apply(lambda x: 1 if i in x else 0)
train["day_2s"] = train["detail"].apply(lambda x: 0 if 28 and 27 in x else 1)
train["day_3s"] = train["detail"].apply(lambda x: 0 if 28 and 27 and 26 in x else 1)
train["day_4s"] = train["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 in x else 1)
train["day_5s"] = train["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 in x else 1)
train["day_6s"] = train["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 and 23 in x else 1)
train["day_7s"] = train["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 and 23 and 22 in x else 1)
train.drop('detail', axis=1, inplace=True)

day_sum_1 = test1['day_sum_detail']
day_sum_1 = day_sum_1.apply(lambda x: x.split(","))
for i in range (len(day_sum_1)):
    day_sum_1[i] = list(map(int, day_sum_1[i]))
test1['detail'] = day_sum_1
for i in range (1, 29):
    test1['day_'+str(i)] = test1['detail'].apply(lambda x: 1 if i in x else 0)
test1["day_2s"] = test1["detail"].apply(lambda x: 0 if 28 and 27 in x else 1)
test1["day_3s"] = test1["detail"].apply(lambda x: 0 if 28 and 27 and 26 in x else 1)
test1["day_4s"] = test1["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 in x else 1)
test1["day_5s"] = test1["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 in x else 1)
test1["day_6s"] = test1["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 and 23 in x else 1)
test1["day_7s"] = test1["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 and 23 and 22 in x else 1)
test1.drop('detail', axis=1, inplace=True)

day_sum_2 = test2['day_sum_detail']
day_sum_2 = day_sum_2.apply(lambda x: x.split(","))
for i in range (len(day_sum_2)):
    day_sum_2[i] = list(map(int, day_sum_2[i]))
test2['detail'] = day_sum_2
for i in range (1, 29):
    test2['day_'+str(i)] = test2['detail'].apply(lambda x: 1 if i in x else 0)
test2["day_2s"] = test2["detail"].apply(lambda x: 0 if 28 and 27 in x else 1)
test2["day_3s"] = test2["detail"].apply(lambda x: 0 if 28 and 27 and 26 in x else 1)
test2["day_4s"] = test2["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 in x else 1)
test2["day_5s"] = test2["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 in x else 1)
test2["day_6s"] = test2["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 and 23 in x else 1)
test2["day_7s"] = test2["detail"].apply(lambda x: 0 if 28 and 27 and 26 and 25 and 24 and 23 and 22 in x else 1)
test2.drop('detail', axis=1, inplace=True)

# drop feature
drop_list = ['charid_sum_detail', 'server_sum_detail','day_sum_detail','kind_sum_detail','kind_sum_count',
             'class_concat_detail','charid_concat_detail_combat', 'server_concat_detail_combat', 'level_concat_detail',
             'day_concat_detail_combat','kind_concat_detail_combat', 'kind_concat_count_combat', 
            'day_concat_detail_payment', 'day_concat_count_payment','kind_concat_detail_payment', 'kind_concat_count_payment', 
            'day_concat_detail_pledge', 'charid_concat_detail_pledge','server_concat_detail_pledge',
             'pledgeid_concat_detail_pledge','kind_concat_detail_pledge','kind_concat_count_pledge_pledge',
            'day_sum_detail_trade','server_sum_detail_trade', 'char_sum_detail_trade', 'trade_accid_detail', 'day_payment','kind']
train.drop(drop_list, axis = 1, inplace = True)
test1.drop(drop_list, axis = 1, inplace = True)
test2.drop(drop_list, axis = 1, inplace = True)

def getTrainLabelData():
    return train_label

def getTrainData():
    return train

def getTest1Data():
    return test1

def getTest2Data():
    return test2
