from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from self_def import count_interval,text_setting,offerPlot,funnel

font = text_setting()

engine = create_engine("mysql+pymysql://root:password@localhost/project_app_data")

sql_record_info ='''
SELECT member_id,offer_id,offer_type,received_time,viewed_time,completed_time,(duration*24+ offer_record.received_time) AS duration_end_hour
FROM
	offer_record
JOIN 
	offer_info
ON
	offer_record.offer_id= offer_info.id
ORDER BY
	member_id
'''
#載入會員跟期相關的offer資訊
record_info = pd.read_sql(sql =sql_record_info, con= engine)
#將recieved_time從小時轉變為時間利於圖表呈現
record_info['received_time(day)'] =record_info['received_time'].apply(lambda x : int( x /24 + 1))

#有效觀看(需在時限前view 且比completed_time小確保是看了後才下單) : 
#有效觀看篩選條件:觀看時間<時限 & 觀看時間 != -1(表示無此行為)  & (有購買之下，觀看時間<購買時間|無購買之下，觀看時間>購買時間)
mask_viewed_time = ((record_info['viewed_time'] <= record_info['duration_end_hour']) & (record_info['viewed_time'] != -1)) & (((record_info['completed_time'] != -1) & (record_info['viewed_time'] <= record_info['completed_time'] )) |((record_info['completed_time'] == -1) & (record_info['viewed_time'] > record_info['completed_time'] ))) 
#將有效觀看由小時轉為天
record_info['effective_viewed_time(day)'] =record_info[mask_viewed_time ]['viewed_time'].apply(lambda x : int( x /24 + 1))

#有效交易(比view_time大確保是看了後才下單) -->只有discount +bogo 才有對應的交易
#有效觀看篩選條件:有效觀看時間>0 & 
mask_completed_time  = (record_info['effective_viewed_time(day)'] > 0 ) & (record_info['completed_time'] != -1)   
#將有效購買由小時轉為天
record_info['effective_completed_time(day)'] = record_info[mask_completed_time]['completed_time'].apply(lambda x : int( x /24 + 1))


#計算30天走期內發送幾筆offer
informational_count = count_interval(1,31,record_info,'received_time(day)','informational')
bogo_count = count_interval(1,31,record_info,'received_time(day)','bogo')
discount_count = count_interval(1,31,record_info,'received_time(day)','discount')

#計算30天走期內有效點閱幾筆offer
informational_count_view = count_interval(1,31,record_info,'effective_viewed_time(day)','informational')
bogo_count_view = count_interval(1,31,record_info,'effective_viewed_time(day)','bogo')
discount_count_view = count_interval(1,31,record_info,'effective_viewed_time(day)','discount')

#計算30天走期內有完成幾筆offer
bogo_count_complete = count_interval(1,31,record_info,'effective_completed_time(day)','bogo')
discount_count_complete = count_interval(1,31,record_info,'effective_completed_time(day)','discount')

#檢查數據
#print(sum(informational_count_complete )+sum(bogo_count_complete)+sum(discount_count_complete))
#print(record_info['effective_completed_time(day)'].notnull().sum())

#繪製informational走期折線圖
offerPlot(informational_count,informational_count_view)
plt.legend(prop={'size': 17})
plt.xticks(range(1,31,1))
plt.yticks(range(0,3000,500))
plt.ylim(0,3000)
plt.grid(color = '#A2D2DF',zorder=0,alpha=0.5)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('時間(第X天)',size=20)
plt.ylabel('人次',size=20,rotation=0,ha='right')
plt.title('曝光訊息-走期v.s.人次',size=20)
plt.show()


#繪製bogo走期折線圖
offerPlot(bogo_count,bogo_count_view)
#比informational多completed的線
time = np.array([i for i in range(1,31)])
plt.plot(time,bogo_count_complete,color='#387F39',label='點數接收',linewidth=2)
for x, y in zip(time, bogo_count_complete):
    if y != 0:
        plt.text(x, y , f'{y}', ha='center', va='bottom', fontsize=14, color='#387F39')

plt.legend(prop={'size': 17})
plt.xticks(range(1,31,1))
plt.yticks(range(0,5500,500))
plt.ylim(0,5500)
plt.grid(color = '#A2D2DF',zorder=0,alpha=0.5)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('活動時間(第X天)',size=20)
plt.ylabel('人次',size=20,rotation=0,ha='right')
plt.title('促銷活動-走期v.s.人次',size=20)
plt.show()


#繪製discount走期折線圖
offerPlot(discount_count,discount_count_view)
#比informational多completed的線
time = np.array([i for i in range(1,31)])
plt.plot(time,discount_count_complete,color='#387F39',label='點數接收',linewidth=2)
for x, y in zip(time, discount_count_complete):
    if y != 0:
        plt.text(x, y , f'{y}', ha='center', va='bottom', fontsize=14, color='#387F39')

plt.legend(prop={'size': 17})
plt.xticks(range(1,31,1))
plt.yticks(range(0,5500,500))
plt.ylim(0,5500)
plt.grid(color = '#A2D2DF',zorder=0,alpha=0.5)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('活動時間(第X天)',size=20)
plt.ylabel('人次',size=20,rotation=0,ha='right')
plt.title('折扣活動-走期v.s.人次',size=20)
plt.show()

#%%
#有效交易資料
sql_effective_offer_transaction ='''
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
'''
#y資料檢查-比對transaction_time與completed_time差額原因:132同筆交易同時滿足2個以上同樣的offer_completed/print((record_info['effective_completed_time(day)']>=0).sum())


#有效交易資料(與offer_completed有關聯的)，因transaction 為該走期發生的交易但不是每筆交易都跟offer有關
effective_offer_transaction = pd.read_sql(sql = sql_effective_offer_transaction, con= engine)
#取出促銷、折扣欄位
duration_info =effective_offer_transaction[['offer_id','duration']].drop_duplicates()
#因曝光欄位沒有產生交易，故另外加入欄位中
duration_info.loc[8] =['5a8bc65990b245e5a138643cd4eb9837',3]
duration_info.loc[9] =['3f207df678b143eea3cee63160fa8bed',4]


#將活動訊息join offer_record紀錄表
record_info = pd.merge(record_info,duration_info,on ='offer_id',how ='left') 
#繪製折扣漏斗圖
funnel(record_info, effective_offer_transaction,'discount',"折扣")
funnel(record_info, effective_offer_transaction,'bogo',"促銷")

#%%
#informational
labels = ["訊息點閱","訊息接收"]

 #篩選informational的有效點閱
mask_info_effective_view = (record_info['offer_type'] == 'informational') & (record_info['effective_viewed_time(day)'] >= 0)
#篩選informational的有效購買
mask_info_effective_complete = (record_info['offer_type'] == 'informational') & (record_info['effective_completed_time(day)'] >= 0)
#有效點閱的筆數
effective_view_count =record_info[mask_info_effective_view].shape[0]
#將informational的相關人數放到list[發送,有效觀看,有效購買]
person_amount =[(record_info['offer_type'] == 'informational').sum(),effective_view_count]

#有效觀看筆例
view_percent = effective_view_count / person_amount[0] *100
#平均有效點閱時間
avg_view_hour = sum((record_info[mask_info_effective_view]['viewed_time']) -  (record_info[mask_info_effective_view]['received_time'])) /effective_view_count 
#有效觀看在時限內的平均落點
avg_view_duration_percent = sum(((record_info[mask_info_effective_view]['viewed_time']) -  (record_info[mask_info_effective_view]['received_time'])) / (record_info[mask_info_effective_view]['duration']*24)) /effective_view_count *100

#繪製每層漏斗圖
colors = ['#F7A4A4',"#ADA2FF"]
fig = plt.figure(figsize=(10,8))
plt.fill_betweenx(y=[3.85, 4.6], x1=[13,15], x2=[5,3], color=colors[1], edgecolor="#aeb6bf", linewidth=3);
plt.fill_betweenx(y=[2.2, 3.8], x1=[9,13], x2=[9,5], color=colors[0], edgecolor="#aeb6bf", linewidth=3);


plt.xticks([],[]);
plt.yticks([3.2,4.2], labels,fontsize = 18);

plt.text(9, 4.2, f'人次:{person_amount[0]}', fontsize=20, color="black", ha="center");
plt.text(9, 2.9, f'點閱率:{view_percent:.2f}% ({person_amount[1]}人次) \n 平均點閱:\n{avg_view_hour:.2f}/小時 \n 平均時限(%):\n{avg_view_duration_percent:.2f}%' , fontsize=20, color="black", ha="center");

plt.title("曝光活動漏斗", loc="center", fontsize=25);