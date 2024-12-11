from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import pandas as pd
from self_def import text_setting,plotScatter,scatterText

font = text_setting()
engine = create_engine("mysql+pymysql://root:password@localhost/project_app_data")
sql_select = '''
SELECT * FROM offer_info
'''
offer_info = pd.read_sql(sql = sql_select,con = engine)


#繪製最低銷售門檻VS回饋點數的散佈圖
plt.figure(figsize=(10,5))
plotScatter(offer_info, 'difficulty', 'reward', 0.1)
plt.legend(loc="upper left")

#座標點標記,因有多個點重合，文字內容需另外寫
for i, row in offer_info[offer_info["offer_type"] == 'informational'].iterrows():
    times = offer_info[(offer_info["offer_type"] == 'informational') &  ( offer_info['difficulty']==row['difficulty']) & (offer_info['reward']==row['reward'] )].shape[0] 
    plt.text(row['difficulty'], row['reward']+0.3, f"({row['difficulty']}, {row['reward']})*{times} 個", ha='center', va='bottom', fontsize=10)
for i, row in offer_info[offer_info["offer_type"] == 'discount'].iterrows():
    times = offer_info[(offer_info["offer_type"] == 'discount') &  ( offer_info['difficulty']==row['difficulty']) & (offer_info['reward']==row['reward'] )].shape[0] 
    plt.text(row['difficulty'], row['reward']+0.3, f"({row['difficulty']}, {row['reward']})*{times} 個", ha='center', va='bottom', fontsize=10)
for i, row in offer_info[offer_info["offer_type"] == 'bogo'].iterrows():
    times = offer_info[(offer_info["offer_type"] == 'bogo') &  ( offer_info['difficulty']==row['difficulty']) & (offer_info['reward']==row['reward'] )].shape[0] 
    plt.text(row['difficulty'], row['reward']+0.3, f"({row['difficulty']}, {row['reward']})*{times} 個", ha='center', va='bottom', fontsize=10)


plt.xlabel("最低花費金額(元)",size= 14)
plt.ylabel("獎勵點數(點)",rotation=0,ha='right',size= 14)
plt.title("獎勵點數v.s.最低花費金額",size= 14)
plt.grid(color = '#EEF7FF',zorder=0)

plt.xticks(range(0, 22, 2))
plt.yticks(range(0, 12, 1))
plt.ylim(0, 11)
plt.xlim(0, 22)
plt.show()

#%%


#繪製時限VS回饋點數的散佈圖
plt.figure(figsize=(6,6))
plotScatter(offer_info, 'duration', 'reward', 0.05)
plt.legend(loc="upper left")

#座標點標記
scatterText(offer_info,'duration','reward')

plt.xlabel("活動時長(天)",size= 14)
plt.ylabel("獎勵點數(點)",rotation=0,ha='right',size= 14) 
plt.title("獎勵點數v.s.活動時長",size= 14)
plt.grid(color = '#EEF7FF',zorder=0)

plt.xticks(range(0, 22, 2))
plt.yticks(range(0, 12, 1))
plt.ylim(0, 11)
plt.xlim(0, 11)
plt.show()


#%%

#繪製最低花費VS時限散佈圖
plt.figure(figsize=(10,5))
plotScatter(offer_info, 'difficulty', 'duration', 0.1)
plt.legend()

#座標點標記
scatterText(offer_info, 'difficulty', 'duration')

plt.xlabel("最低花費金額(元)",size= 14)
plt.ylabel("活動時長(天)",rotation=0,ha='right',size= 14)
plt.title("活動時長v.s.最低花費金額",size= 14)
plt.grid(color = '#EEF7FF',zorder=0)

plt.xticks(range(0, 22, 2))
plt.yticks(range(0, 12, 1))
plt.ylim(0, 11)
plt.xlim(0, 22)
plt.show()


#%%

#方案傳送管道統計長條圖(只有10筆資料自己人工計數)
channel_list = ['郵件', '網站', '簡訊', '社群']
channel_count=[10,8,9,6] 
plt.figure(figsize=(4,6))
plt.bar(channel_list,channel_count,zorder=1,width=0.5, color='#B4BDFF')
plt.xlabel("管道",size= 12)
plt.ylabel("方案數量",rotation=0,ha='right',size= 12)
plt.title("透過特定管道傳送活動訊息的方案數量",size= 12)
for i,j in zip(channel_list,channel_count):
    plt.text(i,j +0.1,j, ha='center', va='bottom', fontsize=12)

#%%

#性別比例圓餅圖
import numpy as np
from self_def import percent_with_actualcount
engine_meber_info = create_engine("mysql+pymysql://root:password@localhost/project_app_data")
sql_meber_info  = '''
SELECT * FROM member_info
'''
member_info = pd.read_sql(sql = sql_meber_info,con = engine)
gender_group = member_info.groupby('gender')
#計算每組分群的筆數
gender_counts = gender_group.size()


#因groupby會把NA值忽略，未呈現真實資料，把NA的數量統計
gender_counts.loc['無資料'] = member_info['gender'].isna().sum()
#其他性別欄位數值最少希望順序排最後所以新增一個其他欄位讓他排在最後，再將原本的其他性別欄位刪除
gender_counts.loc['其他'] = gender_counts['O']
gender_counts = gender_counts.drop(labels=['O'])
colors =['#FF8080','#FFCF96','#B4BDFF','#9ED2BE']
labels = ['女','男','其他','無資料']
explode=[0.02,0.02,0.02,0.03]

plt.figure(figsize=(4,4))
#autopct 要呈現筆例外也要呈現實際數值，邏輯寫在函數中
plt.pie(gender_counts,labels= labels,colors = colors,shadow= True ,autopct=lambda pct: percent_with_actualcount(gender_counts, pct),startangle=20,explode=explode,textprops={'fontsize': 9})
plt.title('性別比例')

#%%

#年齡區間直方圖
#計算系統預設值的數量(12%)
age_defalt_count = member_info[member_info['age'] == 118]['age'].count()
#計算100歲以上非118(系統預設的)數量:17筆)
age_over_100_not_defalut = member_info[member_info['age'] >= 100]['age'].count()-age_defalt_count
age_info = member_info[member_info['age'] < 100]['age']  #14808筆/
age_info.describe()
'''
count    14808.000000
mean        54.340829
std         17.323921
min         18.000000
25%         42.000000
50%         55.000000
75%         66.000000
max         99.000000
'''

bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# 計算每個區間的數量和邊界
counts, edges = np.histogram(age_info, bins=bins)
# 計算百分比
percentages = counts / counts.sum() * 100

# 計算每個區間的中心位置作為條形的 y 軸位置，起始點+該區間格的一半即為區間中點
bin_centers = edges[:-1] + np.diff(edges) / 2
plt.figure(figsize=(15,16))
plt.hist(age_info,bins= bins, color='#B4BDFF', edgecolor='#5C2FC2', rwidth=0.8,zorder=2)

# #年齡區間兩側數值作為X軸文字
plt.xticks(bin_centers, [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)],size=14) 
plt.yticks(size=14) 

#直方圖文字文字(數值+總體百分比)
for count, percentage, center in zip(counts, percentages, bin_centers):
    plt.text(center, count + 2, f'{count} \n ({percentage:.1f}%)', ha='center', va='bottom',zorder =3,size =14)
plt.grid(color = '#D2E0FB', zorder=1)
plt.title('人數v.s.年齡區間',size=18)
plt.xlabel('年齡區間',size=18)
plt.ylabel('人數',rotation=0,ha='right',size=18)
plt.show()

#%%

#年齡區間直方圖
salary_info = member_info[member_info['age'] < 100]['income']
print(salary_info.describe())

'''
count     14808.000000
mean      65394.313884
std       21595.072904
min       30000.000000
25%       49000.000000
50%       64000.000000
75%       80000.000000
max      120000.000000
'''

bins_salary = [30000, 40000,50000, 60000,70000, 80000, 90000, 100000, 110000, 120000,130000]
bins_salary_label = ['3萬', '4萬','5萬', '6萬','7萬', '8萬', '9萬', '10萬', '11萬', '12萬','13萬']
# 計算每個區間的數量和邊界
counts_salary, edges_salary = np.histogram(salary_info, bins=bins_salary)
# 計算百分比
percentages_salary = counts_salary / counts_salary.sum() * 100
# 計算每個區間的中心位置作為條形的 y 軸位置，起始點+該區間格的一半即為區間中點
bin_centers_salary = edges_salary[:-1] + np.diff(edges_salary) / 2

plt.figure(figsize=(18,16))
# Y軸的年齡區間標籤
plt.hist(salary_info,bins= bins_salary, color='#B4BDFF', edgecolor='#5C2FC2', rwidth=0.8,zorder=2)
#年齡區間兩側數值作為X軸該文字
plt.xticks(bin_centers_salary, [f'{bins_salary_label[i]}-{bins_salary_label[i+1]}' for i in range(len(bins_salary_label)-1)]) 

#直方圖文字文字(數值+總體百分比)
for count_salary, percentage_salary, center_salary in zip(counts_salary, percentages_salary, bin_centers_salary):
    plt.text(center_salary, count_salary + 2, f'{count_salary} \n ({percentage_salary:.1f}%)', ha='center', va='bottom',zorder =3,size=16)

plt.grid(color = '#D2E0FB', zorder=1)
plt.yticks(size=18)
plt.xticks(size=18)
plt.xlabel('收入區間',size=20)
plt.ylabel('人數',rotation=0,ha='right',size=20)
plt.title('人數v.s.收入區間',size=20)
plt.show()

#%%
from datetime import datetime
#從2018/9/1往前回推入會時間
start_date = datetime.strptime('2018/09/01','%Y/%m/%d').date()
member_years = member_info['became_member_on']
years_till_now= (start_date - member_years)
#將入會天數轉換為年並四捨五入到整數
years_till_now = years_till_now.apply(lambda x: round((x.days/365),0))
print(years_till_now.describe())
'''
count    17000.000000
mean         1.417671
std          1.126659
min          0.000000
25%          0.570000
50%          0.980000
75%          2.170000
max          4.990000
'''
bins_years = [0,1,2,3,4,5]
# 計算每個區間的數量和邊界
counts_years, edges_years = np.histogram(years_till_now, bins=bins_years)
# 計算百分比
percentages_years = counts_years / counts_years.sum() * 100
# 計算每個區間的中心位置作為條形的 y 軸位置，起始點+該區間格的一半即為區間中點
bin_centers_years = edges_years[:-1] + np.diff(edges_years) / 2
plt.figure(figsize=(6,11))
plt.hist(years_till_now,bins= bins_years, color='#B4BDFF', edgecolor='#5C2FC2', rwidth=0.8,zorder=2)
plt.xticks(bin_centers_years, [f'{bins_years[i]}-{bins_years[i+1]}年' for i in range(len(bins_years)-1)])  # Y軸的年齡區間標籤
for count_years, percentage_years, center_years in zip(counts_years, percentages_years, bin_centers_years):
    plt.text(center_years, count_years + 2, f'{count_years} \n ({percentage_years:.1f}%)', ha='center', va='bottom',zorder =3)
plt.grid(color = '#D2E0FB', zorder=1)
plt.title('人數v.s.入會時間(年)')
plt.xlabel('入會時間(年)')
plt.ylabel('人數',rotation=0,ha='right')
plt.show()

#%%
#查看下兩年以下男性/女性的入會人數(與預測社群廣告推撥圖表比對下)
member_info['years_till_now'] =years_till_now
print(((member_info['gender'] == 'M') & (member_info['years_till_now'] <2)).sum())