from sklearn import preprocessing
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
from self_def import text_setting, process_group, purchase_percentage, decisionTree, neighbor,create_pairplot
import numpy as np


engine = create_engine(
    "mysql+pymysql://root:password@localhost/project_app_data")

# 將會員訊息、offer訊息、offer紀錄合併撈出來
sql_all_info = '''
SELECT 
	member_id,offer_id,offer_type,channels,duration,difficulty,reward,gender,age,became_member_on,income,received_time,viewed_time,completed_time
FROM 
	offer_record
LEFT JOIN
	member_info
on
	member_info.id = offer_record.member_id
LEFT JOIN
	offer_info
on
	offer_info.id = offer_record.offer_id
    '''
all_info = pd.read_sql(sql=sql_all_info, con=engine)

# 欲將客戶行為分類，先建立tag並欲設為-1,並增加check_point欄位作為檢核點預設為0
all_info['tag'] = -1
all_info['check_point'] = 0

# 沒有觀看offer，tag標記為0,並將檢驗點改為1
all_info.loc[all_info['viewed_time'] == -1, 'tag'] = 0
all_info.loc[all_info['viewed_time'] == -1, 'check_point'] = 1

# 時限由天轉換為小時+收到offer時間=有效時間截止點
effective_end_time = all_info['duration']*24 + all_info['received_time']
# 分別filter時限內觀看、時限內觀看
mask_view_out_duration = (all_info['viewed_time'] > effective_end_time) & (
    all_info['check_point'] != 1)
mask_view_in_duration = (all_info['viewed_time'] <= effective_end_time) & (
    all_info['check_point'] != 1)

# 期限外有點外有點閱有購買
all_info.loc[mask_view_out_duration & (
    all_info['completed_time'] != - 1), 'tag'] = 10
# 期限外有點閱無購買
all_info.loc[mask_view_out_duration & (
    all_info['completed_time'] == - 1), 'tag'] = 11
# 更改期限外點閱的檢驗點為1
all_info.loc[mask_view_out_duration, 'check_point'] = 1
# 期限內點閱無購買
all_info.loc[mask_view_in_duration & (
    all_info['completed_time'] == -1), 'tag'] = 21
# 期限內點閱，但在點閱前已經購買
all_info.loc[mask_view_in_duration & (all_info['completed_time'] != -1) & (
    all_info['completed_time'] >= all_info['viewed_time']), 'tag'] = 200
# 期限內點閱後才進行購買
all_info.loc[mask_view_in_duration & (all_info['completed_time'] != -1) & (
    all_info['completed_time'] < all_info['viewed_time']), 'tag'] = 201
# 更改期限內點閱的檢驗點為1
all_info.loc[mask_view_in_duration, 'check_point'] = 1


# 以下開始為計算每個tag在每個方案的百分比的相關步驟
# 依照行為類型的tag分組
group_size = all_info.groupby('tag').size()
print(group_size)

# 給定offer_id名稱，並將名稱對應offer_id欄位放上相應名稱
offer_id_name = {'5a8bc65990b245e5a138643cd4eb9837': '曝光一',
                 '3f207df678b143eea3cee63160fa8bed': '曝光二',
                 'fafdcd668e3743c1bb461111dcafc2a4': '折扣一',
                 '2906b810c7d4411798c6938adc9daaa5': '折扣二',
                 '2298d6c36e964ae4a3e7e9706d1fb8c2': '折扣三',
                 '0b1e1539f2cc45b7b9fa7c272da2e1d7': '折扣四',
                 'f19421c1d4aa40978ebb69ca19b0e20d': '促銷一',
                 '9b98b8c7a33c4b65b9aebfe6a799e6d9': '促銷二',
                 'ae264e3637204a6fb9bb56bc8210ddfd': '促銷三',
                 '4d5c57ea9a6940dd891ad53e9dbe8da0': '促銷四'}

all_info['offer_name'] = ''
for i in range(all_info.shape[0]):
    all_info.loc[i, 'offer_name'] = offer_id_name[all_info.loc[i, 'offer_id']]

# 群組後的index會變成群組欄位，重新設定index並給予群組後的數值欄位名稱
group_offer_id = all_info.groupby(
    ['offer_name', 'tag']).size().reset_index(name='count')
type_name_count = all_info.groupby(
    'offer_name').size().reset_index(name='count')


# 指定offer_id 、tag佔整個同樣offer_id的百分比:計算每個offer_name的總數，並算出百分比/ groupby()[欄位] 只對特定欄位計算/transform('sum') 分組計算總和結果轉換為與原 DataFrame 相同長度的 Series
group_offer_id['count_percent'] = group_offer_id['count'] / \
    group_offer_id.groupby('offer_name')['count'].transform('sum')


# 建立tag X offer_name的PD，預設為0
offer_name = type_name_count['offer_name']
tag_name = group_offer_id['tag'].unique()
offer_name_tag_percent = pd.DataFrame(
    np.zeros((10, 6)), columns=tag_name, index=offer_name)

# 將百分比的資料放到對應的格子裡
for i in offer_name:
    for j in tag_name:
        value = group_offer_id.loc[(group_offer_id['offer_name'] == i) & (
            group_offer_id['tag'] == j), 'count_percent']
        # 如果 value 不為空，則取出值並填入 offer_name_tag_percent，否則保留 0
        if not value.empty:
            offer_name_tag_percent.loc[i, j] = round(value.values[0]*100, 1)

# 做為繪圖的y軸值
bar_value = {
    0: offer_name_tag_percent[0],
    10: offer_name_tag_percent[10],
    11: offer_name_tag_percent[11],
    21: offer_name_tag_percent[21],
    200: offer_name_tag_percent[200],
    201: offer_name_tag_percent[201]
}

color = {
    0: '#8FBDD3',
    10: '#A5B68D',
    11: '#DC8686',
    21: '#BAABDA',
    200: '#D77FA1',
    201: '#FDCEB9'}


font = text_setting()
fig, ax = plt.subplots(figsize=(20, 10))
bottom = np.zeros(10)
order = ['曝光一', '曝光二', '折扣一', '折扣二', '折扣三', '折扣四', '促銷一', '促銷二', '促銷三', '促銷四']
offer_name_tag_percent = offer_name_tag_percent.reindex(order)
for key, value in bar_value.items():
    # y軸的值的順序依照指定順序排列
    value = value.reindex(order)
    percent_bar = ax.bar(value.index, value, width=0.5,
                         label=key, bottom=bottom, color=color[key])
    # 現在y軸的低點+這輪y軸的值做為下一輪y軸值的起始點
    bottom += value
    # 文字標記於數值>1 才呈現
    labels = [f"{float(v)}%" if v > 1 else "" for v in value.to_numpy()]
    # 該組bar的正中間做標記
    ax.bar_label(percent_bar, labels=labels, label_type='center', fontsize=18)


ax.set_title('用戶對於方案行為分析(%)', size=24)
ax.set_ylim(0, 100)
# 設置y軸標籤加上百分比
ax.set_yticklabels([f"{int(y)}%" for y in ax.get_yticks()])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=20)

plt.show()

# %%

# 發現數值與漏斗圖對不上，比對資料差異，發現是計算漏斗圖的判斷式沒加到=，調整後數據一致
'''
check_pd = all_info[(all_info['offer_type'] == 'bogo') &(all_info['tag'] == 200)]
print(check_pd.shape[0])

sql_effective_offer_transaction ="
SELECT 
    Ttable.member_id,
    Ttable.offer_id,
    offer_info.offer_type,
    Ttable.amount,
    completed_time,
    offer_info.difficulty,
    offer_info.reward,
    offer_info.duration
FROM 
    (SELECT 
        offer_record.member_id,
        offer_record.offer_id,
        transaction_record.amount,
        offer_record.completed_time,
        offer_record.viewed_time
     FROM 
        offer_record
     RIGHT JOIN 
        transaction_record 
     ON 
        offer_record.completed_time = transaction_record.transaction_time 
        AND offer_record.member_id = transaction_record.member_id
     WHERE 
        offer_record.completed_time <> -1 AND offer_record.viewed_time <> -1
    ) AS Ttable
LEFT JOIN 
    offer_info
ON 
    Ttable.offer_id = offer_info.id
WHERE  
	Ttable.viewed_time < Ttable.completed_time 
GROUP BY 
	member_id,offer_id,completed_time
ORDER BY 
    member_id;
"


effecitve_pd = pd.read_sql(sql=sql_effective_offer_transaction,con=engine)


non_common_check_pd = pd.merge(check_pd, 
                                effecitve_pd, 
                                how='left', 
                                on=['offer_id', 'member_id', 'completed_time'], 
                                indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

# 3. 找出 effecitve_pd 中沒有一致的列 (right only)
non_common_effective_pd = pd.merge(check_pd, 
                                    effecitve_pd, 
                                    how='right', 
                                    on=['offer_id', 'member_id', 'completed_time'], 
                                    indicator=True).query('_merge == "right_only"').drop('_merge', axis=1)

'''

# %%

# 創立表用來儲存篩選後的不重複名單
all_info_without_duplicate_people = pd.DataFrame(columns=all_info.columns)


# 同會員同方案只取一筆，判斷選取的優先順序如下list(有買-->只有看-->沒有看)
tag_priority = [200, 201, 10, 21, 11, 0]
# 以member_id和offer_id作為唯一值判斷
grouped = all_info.groupby(['member_id', 'offer_id'])
current_index = 0

# 遍歷每一組 (每一組都有相同的 'member_id' 和 'offer_id')
for (member_id, offer_id), group in grouped:
    # 排序tag 標籤，依據tag_priority.index(tag)排序(返回 tag 在 tag_priority 列表中的索引)
    group_sorted = group.sort_values(
        by='tag', key=lambda x: x.apply(lambda tag: tag_priority.index(tag)))
    # 只選擇在排序第一筆(tag 優先級最高的行)，等於去除同方案重複筆數
    first_match_row = group_sorted.iloc[0]
    all_info_without_duplicate_people.loc[current_index] = first_match_row
    current_index += 1

'''check筆數
check_repeat = pd.DataFrame([all_info['offer_id'],all_info['member_id']]).T
check_repeat['count'] = 1
check_repeat_groupby = check_repeat.groupby(['member_id', 'offer_id']).size()
'''
# 計算入會時間
start_date = datetime.strptime('2018/09/01', '%Y/%m/%d').date()
member_years = all_info_without_duplicate_people['became_member_on']
years_till_now = (start_date - member_years)
all_info_without_duplicate_people['years_till_now'] = years_till_now.apply(
    lambda x: round((x.days/365), 0))


# 將標籤轉換為有買-->有看沒買-->沒看
tag_to_label = {0: 0,
                21: 1,
                11: 1,
                200: 2,
                201: 2,
                10: 2
                }

all_info_without_duplicate_people['label'] = -1

# 將新標籤依照tag_t0_label的dict做對應放到tag欄位
for j in range(all_info_without_duplicate_people.shape[0]):
    all_info_without_duplicate_people.loc[j,
                                          'label'] = tag_to_label[all_info_without_duplicate_people.loc[j, "tag"]]

# 選取年齡在100歲以下且非空值的人
mask_member = (all_info_without_duplicate_people['age'].notna()) & (
    all_info_without_duplicate_people['age'] < 100)
unique_record_without_incomplete_member_data = all_info_without_duplicate_people[
    mask_member]


# 篩選channels 欄位包含 'social' 的行
filtered_df = unique_record_without_incomplete_member_data[
    unique_record_without_incomplete_member_data['channels'].str.contains(
        'social', na=False)
]

# 分組為 bogo discount
bogo_with_socail = filtered_df[filtered_df['offer_type'] == 'bogo']
discount_with_socail = filtered_df[filtered_df['offer_type'] == 'discount']


# 刪除多於欄位以及給購買傾向標籤
bogo_unique_member_record = process_group(bogo_with_socail)
discount_unique_member_record = process_group(discount_with_socail)
print(bogo_unique_member_record.shape)
print(discount_unique_member_record.shape)

# %%
# grouby將index改為groupby的欄位，重新設定index並去除多餘的欄位
bogo_tree_X = bogo_unique_member_record[[
    'gender', 'age', 'years_till_now', 'income']].reset_index()
bogo_tree_X.drop(['member_id', 'level_1'], axis=1, inplace=True)
# 發現長的像數字的欄位是object,進行型態轉換
bogo_tree_X = bogo_tree_X.apply(pd.to_numeric, errors='coerce')
bogo_tree_y = bogo_unique_member_record['label']
# 將文字型態的gender轉換為數值表達
bogo_tree_X_encoder = preprocessing.LabelEncoder()
bogo_tree_X['gender'] = bogo_tree_X_encoder.fit_transform(
    bogo_tree_X['gender'])

# 決策樹計算
tree_accuracy_train, tree_accuracy_test, max_tree_accuracy_depth = decisionTree(
    bogo_tree_X, bogo_tree_y)

y_train = tree_accuracy_train
y_test = tree_accuracy_test

X_axis = [i*0.01+0.1 for i in range(86)]

# 將訓練集、測試集、分層數繪製為圖表
fig, ax = plt.subplots(figsize=(30, 12))
ax.plot(X_axis, y_train, color='#DC8686',
        label='訓練集正確率', linewidth=2, zorder=1)
ax.plot(X_axis, y_test, color='#BAABDA', label='測試集正確率', linewidth=2, zorder=1)
# 建立副坐標軸對應的折線圖
ax_plot = ax.twinx()
ax_plot.plot(X_axis, max_tree_accuracy_depth, color='#387F39',
             label='分層數', zorder=0, linestyle='--', linewidth=2)
ax.set_xlabel('測試集比例', size=12)
ax.set_ylabel('正確率', size=12, rotation=0, ha='right')
ax.set_xlim(0.1, 0.96)
ax.set_ylim(0.54, 0.64)
ax.set_xticks(np.arange(0.1, 0.96, 0.02))
ax.set_yticks(np.arange(0.50, 0.64, 0.01))
# ax.set_yticklabels([f"{y*100:.0f}%" for y in ax.get_yticks()])

# 建立副座標軸的刻度及標籤
ax_plot.set_ylabel('分層數', size=12, rotation=0, ha='left')
ax_plot.set_ylim(0, 16)
ax_plot.tick_params(axis='y')

# 取得主、副座標軸的label，合併label於圖例中
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax_plot.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2,
          loc='upper left', prop={'size': 12})
ax.grid(color='#A2D2DF', zorder=0, alpha=0.5)
ax.set_title('決策樹正確率與對應分層數(促銷)', size=14)
plt.show()


# %%
# 近鄰演算法計算
max_accuracy_train, max_accuracy_test, accuracy_n_neighbors = neighbor(
    bogo_tree_X, bogo_tree_y)

y_train = max_accuracy_train
y_test = max_accuracy_test
X_axis = [i*0.01+0.6 for i in range(36)]

# 將訓練集、測試集、分層數繪製為圖表
fig, ax = plt.subplots(figsize=(30, 12))
ax.plot(X_axis, y_train, color='#DC8686',
        label='訓練集正確率', linewidth=2, zorder=1)
ax.plot(X_axis, y_test, color='#BAABDA', label='測試集正確率', linewidth=2, zorder=1)

# 建立副坐標軸對應的折線圖
ax_plot = ax.twinx()
ax_plot.plot(X_axis, accuracy_n_neighbors, color='#387F39',
             label='鄰近個數', zorder=0, linestyle='--', linewidth=2)
ax.set_xlabel('測試集比例', size=12)
ax.set_ylabel('正確率', size=12, rotation=0, ha='right')
ax.set_xlim(0.6, 0.96)
ax.set_ylim(0.50, 0.58)
ax.set_xticks(np.arange(0.6, 0.96, 0.01))
ax.set_yticks(np.arange(0.50, 0.58, 0.01))
# ax.set_yticklabels([f"{y*100:.0f}%" for y in ax.get_yticks()])

# 建立副座標軸的刻度及標籤
ax_plot.set_ylabel('取鄰近X個', size=12, rotation=0, ha='left')
ax_plot.set_ylim(20, 100)
ax_plot.tick_params(axis='y')

# 取得主、副座標軸的label，合併label於圖例中
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax_plot.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2,
          loc='upper left', prop={'size': 12})
ax.grid(color='#A2D2DF', zorder=0, alpha=0.5)
ax.set_title('近鄰演算正確率與對應選取鄰近個數(促銷)', size=14)
plt.show()


# %%

# grouby將index改為groupby的欄位，重新設定index並去除多餘的欄位
discount_tree_X = discount_unique_member_record[[
    'gender', 'age', 'years_till_now', 'income']].reset_index()
discount_tree_X.drop(['member_id', 'level_1'], axis=1, inplace=True)
# 發現長的像數字的欄位是object,進行型態轉換
discount_tree_X = discount_tree_X.apply(pd.to_numeric, errors='coerce')
discount_tree_y = discount_unique_member_record['label']
# 將文字型態的gender轉換為數值表達
discount_tree_X_encoder = preprocessing.LabelEncoder()
discount_tree_X['gender'] = discount_tree_X_encoder.fit_transform(
    discount_tree_X['gender'])

# 決策樹計算
tree_accuracy_train, tree_accuracy_test, max_tree_accuracy_depth = decisionTree(
    discount_tree_X, discount_tree_y)
y_train = tree_accuracy_train
y_test = tree_accuracy_test
X_axis = [i*0.01+0.1 for i in range(86)]

# 將訓練集、測試集、分層數繪製為圖表
fig, ax = plt.subplots(figsize=(30, 12))
ax.plot(X_axis, y_train, color='#DC8686',
        label='訓練集正確率', linewidth=2, zorder=1)
ax.plot(X_axis, y_test, color='#BAABDA', label='測試集正確率', linewidth=2, zorder=1)

# 建立副坐標軸對應的折線圖
ax_plot = ax.twinx()
ax_plot.plot(X_axis, max_tree_accuracy_depth, color='#387F39',
             label='分層數', zorder=0, linestyle='--', linewidth=2)
ax.set_xlabel('測試集比例', size=12)
ax.set_ylabel('正確率', size=12, rotation=0, ha='right')
ax.set_xlim(0.1, 0.96)
ax.set_ylim(0.71, 0.75)
ax.set_xticks(np.arange(0.1, 0.96, 0.02))
ax.set_yticks(np.arange(0.71, 0.75, 0.005))
# ax.set_yticklabels([f"{y*100:.0f}%" for y in ax.get_yticks()])

# 建立副座標軸的刻度及標籤
ax_plot.set_ylabel('分層數', size=12, rotation=0, ha='left')
ax_plot.set_ylim(0, 12)
ax_plot.tick_params(axis='y')

# 取得主、副座標軸的label，合併label於圖例中
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax_plot.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2,
          loc='upper left', prop={'size': 12})
ax.grid(color='#A2D2DF', zorder=0, alpha=0.5)
ax.set_title('決策樹正確率與對應分層數(折扣)', size=14)
plt.show()


# %%
max_accuracy_train, max_accuracy_test, accuracy_n_neighbors = neighbor(
    discount_tree_X, discount_tree_y)

y_train = max_accuracy_train
y_test = max_accuracy_test
X_axis = [i*0.01+0.6 for i in range(36)]

# 建立副坐標軸對應的折線圖
fig, ax = plt.subplots(figsize=(30, 12))
ax.plot(X_axis, y_train, color='#DC8686',
        label='訓練集正確率', linewidth=2, zorder=1)
ax.plot(X_axis, y_test, color='#BAABDA', label='測試集正確率', linewidth=2, zorder=1)
ax_plot = ax.twinx()
ax_plot.plot(X_axis, accuracy_n_neighbors, color='#387F39',
             label='鄰近個數', zorder=0, linestyle='--', linewidth=2)
ax.set_xlabel('測試集比例', size=12)
ax.set_ylabel('正確率', size=12, rotation=0, ha='right')
ax.set_xlim(0.6, 0.96)
ax.set_ylim(0.6, 0.75)
ax.set_xticks(np.arange(0.6, 0.96, 0.01))
ax.set_yticks(np.arange(0.6, 0.75, 0.01))
# ax.set_yticklabels([f"{y*100:.0f}%" for y in ax.get_yticks()])

# 建立副座標軸的刻度及標籤
ax_plot.set_ylabel('取鄰近X個', size=12, rotation=0, ha='left')
ax_plot.set_ylim(15, 55)
ax_plot.tick_params(axis='y')

# 取得主、副座標軸的label，合併label於圖例中
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax_plot.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2,
          loc='upper left', prop={'size': 12})
ax.grid(color='#A2D2DF', zorder=0, alpha=0.5)
ax.set_title('近鄰演算正確率與對應選取鄰近個數(折扣)', size=14)
plt.show()

# %%
# 取兩年內入會的'age','gender','label'欄位
bogo_unique_member_newcomer = bogo_unique_member_record[bogo_unique_member_record['years_till_now'] < 2][[
    'age', 'gender', 'label']]
discount_unique_member_newcomer = discount_unique_member_record[discount_unique_member_record['years_till_now'] < 2][[
    'age', 'gender', 'label']]
# 年齡分割表
bins_age = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 繪製兩年內入會的性別X年齡的購買傾向圖(%)、購買人數
purchase_percentage(bins_age, bogo_unique_member_newcomer, "促銷")
purchase_percentage(bins_age, discount_unique_member_newcomer, "折扣")

# %%

# print(((discount_unique_member_record['gender'] == 'M')  &  (discount_unique_member_record['years_till_now'] <2)).sum())

# print(((bogo_unique_member_record['gender'] == 'F')  &  (bogo_unique_member_record['years_till_now'] <2)).sum())

# %%
from pandas.plotting import parallel_coordinates
#取bogo有用社群管道發送的不重複會員
bogo_df = bogo_unique_member_record.drop(['member_id'], axis=1)
#重新設立index
bogo_df =bogo_df.reset_index()
#刪除不要欄位
bogo_df.drop(['member_id', 'level_1'], axis=1, inplace=True)
#取出應變數
bogo_df_label = bogo_df['label'] 
#取出自變數
bogo_df = bogo_df[['gender', 'age', 'years_till_now', 'income']]
#將gender轉換為數值資料
bogo_df_encoder = preprocessing.LabelEncoder()
bogo_df['gender'] = bogo_df_encoder.fit_transform(bogo_df['gender'])
#將自變數的type都轉成int
bogo_df = bogo_df.apply(pd.to_numeric, downcast='integer')

#對於自變數進行正規化
min_max = preprocessing.MinMaxScaler()
bogo_df_min_max = pd.DataFrame(min_max.fit_transform(bogo_df), columns=bogo_df.columns)
#將應變數與自變數欄位連接起來
bogo_parallel_df = pd.concat([bogo_df_min_max,bogo_df_label], axis=1)

'''多維度水平觀測，筆數太多效果不好
#bogo_pc= parallel_coordinates(bogo_parallel_df, 'label', color=('#8FBDD3', '#A5B68D','#DC8686','#BAABDA','#D77FA1'))
bogo_pc= parallel_coordinates(bogo_parallel_df[bogo_parallel_df['label'] == 20], 'label', color=('#8FBDD3'))
bogo_pc= parallel_coordinates(bogo_parallel_df[bogo_parallel_df['label'] == 21], 'label', color=('#A5B68D'))
bogo_pc= parallel_coordinates(bogo_parallel_df[bogo_parallel_df['label'] == 10], 'label', color=('#DC8686'))
bogo_pc= parallel_coordinates(bogo_parallel_df[bogo_parallel_df['label'] == 11], 'label', color=('#BAABDA'))
'''
#將欄位改名稱，圖表label輸出就不用再改
bogo_parallel_df.columns =['性別','年齡','入會時長','收入','標籤']

#依據不同客群(label)畫出散佈圖矩陣
create_pairplot(bogo_parallel_df, 20, '#8FBDD3', '(客群:有時會買)')
create_pairplot(bogo_parallel_df, 21, '#A5B68D', '(客群:每次都會買)')
create_pairplot(bogo_parallel_df, 10, '#DC8686', '(客群:有時會看)')
create_pairplot(bogo_parallel_df, 11, '#BAABDA', '(客群:每次都會看)')
create_pairplot(bogo_parallel_df, 0, '#D77FA1', '(客群:未曾觀看)')


#%%

#取discount有用社群管道發送的不重複會員
discount_df = discount_unique_member_record.drop(['member_id'], axis=1)
#重新設立index
discount_df =discount_df.reset_index()
#刪除不要欄位
discount_df.drop(['member_id', 'level_1'], axis=1, inplace=True)
#取出應變數
discount_df_label = discount_df['label'] 
#取出自變數
discount_df = discount_df[['gender', 'age', 'years_till_now', 'income']]
#將gender轉換為數值資料
discount_df_encoder = preprocessing.LabelEncoder()
discount_df['gender'] = discount_df_encoder.fit_transform(discount_df['gender'])
#將自變數的type都轉成int
discount_df = discount_df.apply(pd.to_numeric, downcast='integer')


#對於自變數進行正規化
min_max = preprocessing.MinMaxScaler()
discount_df_min_max = pd.DataFrame(min_max.fit_transform(discount_df), columns=discount_df.columns)
#將應變數與自變數欄位連接起來
discount_parallel_df = pd.concat([discount_df_min_max,discount_df_label], axis=1)

#將欄位改名稱，圖表label輸出就不用再改
discount_parallel_df.columns =['性別','年齡','入會時長','收入','標籤']

#依據不同客群(label)畫出散佈圖矩陣
create_pairplot(discount_parallel_df, 20, '#8FBDD3', '(客群:有時會買)')
create_pairplot(discount_parallel_df, 21, '#A5B68D', '(客群:每次都會買)')
create_pairplot(discount_parallel_df, 10, '#DC8686', '(客群:有時會看)')
create_pairplot(discount_parallel_df, 11, '#BAABDA', '(客群:每次都會看)')
create_pairplot(discount_parallel_df, 0, '#D77FA1', '(客群:未曾觀看)')