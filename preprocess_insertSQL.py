import requests
import pandas as pd
import ast
from create_SQLTable import connect,cursor
from self_def import combine_rows_to_pandas


url = 'https://storage.googleapis.com/kagglesdsdata/datasets/690741/1210253/portfolio.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241029%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241029T132827Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2241a3fb7225bf1a9e1dc122e1af2b66c1e43706a38241b674308d5e9d8d4e8595a8449d0166fdff7d1e618d2982f6d05bb36c48ebb4f38ed523dca4b5a5dadf92cb5f7602c855fae0c9cfdc73ff1ebb61d88584c76ca1b7189831cb7ba1f9be9e2f3f4b70704b1607989ed6d70d97407f595a7e8b7e84e2551e288ef7eed9db57ebf5df0835c09c50720c619311fec6335a30782bf7369c8cf97239d3f5b7f0c0adf591cf52461ab4eac7c8ddf0566cfc37683340907810204c63682a5ac6f812d67935da663987e084787f1031f5f64bf185ed411994ab6f61cea52fda129d883b3ba7ce48708ebc80661c0850dd7f87c2a34a7dc35f144d81b2221b4ca47e'
api_response = requests.get(url)
#該檔案每筆資料都是json格式但沒有用list將json包起來，故先轉string再用換行分割
api_response_text = api_response.text
api_response_list = api_response_text.split('\n')

#先建立要放最終資料的表
offer_info_df =None 
#將每筆資料轉換類別並放入新pd中
for row_index,i in enumerate(api_response_list):
    try:
        #將list中每個長的像json的str轉為json格式
        i = ast.literal_eval(i)
        #channels對應的value為list,用join函數把list中的element以,串接為一個string
        i['channels'] = ','.join(i['channels'])  
        #將每筆json資料逐一放入offer_info_df中
        offer_info_df = combine_rows_to_pandas(i,row_index,offer_info_df)      
    except SyntaxError:
        pass #只有最後一筆有問題(空格)直接pass
           
#將每筆方案資料逐一插入資料庫
for i in offer_info_df.iterrows():
    sql ='''
    INSERT INTO project_app_data.offer_info VALUES('{}','{}','{}','{}','{}','{}')
    '''
    sql = sql.format(i[1]['id'],i[1]['reward'],i[1]['channels'],i[1]['difficulty'],i[1]['duration'],i[1]['offer_type'])
    cursor.execute(sql)
connect.commit()
connect.close()

#%%
import requests
import ast
from sqlalchemy import create_engine
from create_SQLTable import connect
from self_def import combine_rows_to_pandas

with open ('profile.json') as file:
    data = file.read() 
#檔案中的空值為Null,未來要轉換成Json故將Null取代為None,因讀取近來為str先換行分割包成list
data = data.replace('null', 'None') 
data_list = data.split("\n")

offer_info_df = None
count = 0  #計算有幾筆資料、第幾筆資料
error=[]  #放入錯誤資料內容及資料位置
#逐一合併每筆資料成df
for row_index,i in enumerate(data_list):
    try:
        i = ast.literal_eval(i)
        offer_info_df = combine_rows_to_pandas(i, row_index, offer_info_df)
        count+=1
    except SyntaxError:
        #確認是第幾筆出問題以及該筆內容為何
        error.append([count+1,i]) 
        continue
    
#print(error) #發現皆為空行

#print(offer_info_df.info()) #2175Non 在 gender & income

#數據檢查
# 2175 筆gender & income 都是None 且年齡皆為1181,表示118是沒有填寫的系統預設值
#mask = (offer_info_df['gender'].isnull()) & (offer_info_df['income'].isnull()) & (offer_info_df['age'] == 118)
#print(offer_info_df[mask]) 

#became_member_on' 格式都合規
#print((offer_info_df['became_member_on'].str.match('^201[0-9](0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])$')).sum())

#取不重複值:['F' 'M' None 'O']
#print(offer_info_df['gender'].unique())
#print(offer_info_df[offer_info_df['gender'] == 'O']) 

#income格式都合規
#income_noNa = offer_info_df['income'].dropna()
#print(income_noNa.map(str).str.match('^[1-9][0-9]{1,}0{3}$').sum())

#age_noNa = offer_info_df['age'] != 118  
#去除118剩下應該剩下14825 但只有14808，表示有17筆不在10~99歲(5筆 101  12筆100歲)
#print(offer_info_df[age_noNa]['age'].map(str).str.match('^[1-9][0-9]$').sum()) 
#offer_info_df['age_error'] =offer_info_df[age_noNa]['age'].map(str).str.match('^[1-9][0-9]$')
#print(offer_info_df[offer_info_df['age_error'] == False]) 


#id唯一值確認:筆數與不重複一致(17000)-->沒有重複
#print(offer_info_df['id'].value_counts().sum()) 

#日期欄位為str，將型態改為日期
offer_info_df['became_member_on'] = offer_info_df['became_member_on'] .apply(lambda x : pd.to_datetime(x,format='%Y%m%d'))

#將客戶資料表放到DB
engine = create_engine("mysql+pymysql://root:password@localhost/project_app_data")
offer_info_df.to_sql('member_info', con = engine, if_exists ='append', index = False)





#%%
import requests
import ast
from create_SQLTable import connect,cursor
from self_def import combine_rows_to_pandas,input_offer_time
from sqlalchemy import create_engine


with open('transcript.json') as file:
    data = file.read()
data_list = data.split("\n")
#計算第幾筆資料
count = 0
#放置error
error=[]
#預計offer類型放同張表的不同欄位，transaction放另外一張表，所以先把event分成四類用四個容器裝
offer_received = []
offer_viewd = []
offer_completed = []
transaction = []

for i in data_list:
    try: #把四個類型的offer_id 放到各自的list中，有錯誤則放在錯誤容器中
        i = ast.literal_eval(i)
        if i['event'] == 'offer received':
            i['value'] = i['value']['offer id']
            offer_received.append(i)
        elif i['event'] == 'offer viewed':
            i['value'] = i['value']['offer id']
            offer_viewd.append(i)
        elif i['event'] == 'offer completed':
            i["reward"] = i['value']['reward']
            i['value'] = i['value']['offer_id']
            offer_completed.append(i)
        elif i['event'] == 'transaction':
            i['value'] = i['value']['amount']
            transaction.append(i)
        else:
            print(i)
        count+=1
    except SyntaxError:
        #確認是第幾筆出問題以及該筆內容為何
        error.append([count+1,i])
        continue
#print(count)
#print(error) #發現皆為空行(115筆)


'''
#數據檢查
engine = create_engine("mysql+pymysql://root:password@localhost/project_app_data")
sql ='
SELECT id,reward from offer_info 
'
#offer_plan = pd.read_sql(sql, con = engine)
#offer_completed的reward數值與offer_id紀錄的一致
#for i in offer_completed:
#    mask = offer_plan['id'] == i['reward']
#    if (offer_plan[mask]['reward'].values) != (i['reward']):
#        print(i)

#time皆為>=0的數值
#for i,j,k,l in zip (offer_received,offer_viewd,offer_completed,transaction):
#    if i['time'] < 0  or  j['time'] < 0 or k['time'] < 0 or l['time'] < 0:
'''
#合併offer_received的每筆資料到pd
#       print(i,j,k,l)
offer_record = None
for index,content in enumerate(offer_received):
    offer_record = combine_rows_to_pandas(content, index, offer_record)
       
    
#先把viewed的資料
offer_record = pd.DataFrame([offer_record['person'],offer_record['value'],offer_record['time']]).T
offer_record.columns = ['member_id','offer_id','received_time']

#發現同個offer_id會發給同一人多次 
#offer_record['duplicate_check'] = offer_record['member_id'] + '-' + offer_record['offer_id']
#print(offer_record[offer_record['duplicate_check'].duplicated()]['offer_id'].unique())
#把view資料表建立viewed_time、completed_time欄位，以讓同個offer_id產生的相關時間數據在同筆資料中並預設為-1(-1表示該欄位沒資料)
offer_record[['viewed_time','completed_time']] =-1


#欲放入資料的欄位內容是否已經篩過資料，預設為0(未塞入過)，如已經篩入數值會改就不可再篩資料
offer_record['check_column'] = 0
#將view_time篩入offer_view欄位，詳見input_offer_time函數
exception_list_1 = input_offer_time(offer_record,offer_viewd,'viewed_time') #沒有例外資料


#checkview_time欄位篩入是否有異常(沒有篩入卻不等於-1):沒有
#mask = (offer_record['check_column'] != 0) & (offer_record['viewed_time'] == -1)
#print(offer_record[mask])
#print(offer_record[offer_record['viewed_time'] != -1])
#計算view非-1的數量


#將檢測欄位歸0作為下一次篩入資料的檢測欄
offer_record['check_column'] = 0
offer_record['completed_time'] =-1
#completed_time篩入offer_completed欄位，詳見input_offer_time函數
exception_list_2 = input_offer_time(offer_record,offer_completed,'completed_time') 
#return 478筆異常值中抽查幾筆發現是一次交易完成兩個代號一樣的offer_id

#查看有無異常值，並確認位填入completed_time的欄位有哪些
#mask_2 = (offer_record['check_column'] == 0) & (offer_record['completed_time'] != -1)
#print(offer_record[mask_2])
#print(offer_record[offer_record['completed_time'] != -1])

#避免原始offer_record後續處理被弄毀，先複製一份
offer_record_2 = offer_record.copy()


#將478筆篩入還沒被填入completed_time資料填入offer_completed欄位
exception_list_3 = input_offer_time(offer_record_2,exception_list_2,'completed_time') 
#mask_3 = (offer_record_2['check_column'] != 0) & (offer_record_2['completed_time'] == -1)
#print(offer_record[mask_3])
#print(offer_record[offer_record_2['completed_time'] != -1])

#去除檢驗欄位後將整張資料篩入DB
offer_record_2.drop(['check_column'],axis=1,inplace=True)
offer_record_2.to_sql('offer_record', con = engine, if_exists ='append', index = False)
 

#%%

transaction_error=[]
transaction_df =None
transaction_count = 0
for row_index,i in enumerate(transaction):
    transaction_df = combine_rows_to_pandas(i, row_index, transaction_df)
    try:
        transaction_df = combine_rows_to_pandas(i, row_index, transaction_df)
        transaction_count=1
    except SyntaxError:
        #確認是第幾筆出問題以及該筆內容為何
        transaction_error.append([transaction_count+1,i]) 
        continue

#每筆成交資料的類別都是event故drop
transaction_df.drop(['event'],axis=1,inplace=True)
#把欄位名稱與DB一致
transaction_df.columns=['member_id','amount','transaction_time']
transaction_df.to_sql('transaction_record',con =engine,if_exists='append',index = False)