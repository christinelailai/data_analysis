import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn import neighbors
from sklearn import tree
import time
import seaborn as sns

#將每個json資料轉換為pd並與先前的資料合併成新的pd
#參數:每筆json資料,索引值,最終完整合併所有資料的df
def combine_rows_to_pandas(content, index, offer_df ):
    df = pd.DataFrame(content,index=[index])
    if index == 0: #第一個pd不用做合併操作
        offer_df = df
    else:
        offer_df = pd.concat([df,offer_df],axis=0)
    return offer_df

#透過比對member_id offer_id 來確認是否為同筆offer的資訊
#同時加入> received_time來確保沒有異常值篩入、check_colum是否為0確認沒有倍篩入過資料，並於篩入後將'check_column'改為1
#如果有上述異常狀況都先篩入error_list中後續印出來確認
#參數:數據來源表,欲篩入新表的哪個欄位,數據來源表的取值的欄位名稱
def input_offer_time(offer_record,offer_list,put_in_column):
    exception_list = []
    for i in offer_list:
        try:
            mask = (offer_record['member_id'] == i['person']) & (offer_record['offer_id'] == i['value']) & (offer_record['received_time'] <= i['time'])  # & (offer_record['check_column'] == 0 )  例外處理的時候再加入此條件   
            #可能會有同一個complted_time的member_id&person_id對應到不同兩筆資料(表示同個方案發送多次給同一人)，優先篩入與view_timet時間差距最小的
            target_index = (i['time'] - offer_record[mask]['received_time']).idxmin()     
            if offer_record.loc[target_index,'check_column'] == 0:           
                offer_record.loc[target_index,put_in_column] = i['time'] 
                offer_record.loc[target_index,'check_column'] = 1
            else:
                exception_list.append(i)
        except Exception as e:
            print("錯誤:",e,"\n",i)
            continue
    return exception_list      
            

#繪製三個類型行銷活動的散佈圖 
#參數:數據來源表,篩選欄位一,篩選欄位二,座標點上移多少(為了圖表中呈現好被看到)
def plotScatter(data,feature1,feature2,upper):
    plt.scatter(data[data["offer_type"] == 'informational'][feature1] + upper, data[data["offer_type"] == 'informational'][feature2] + upper,c='#F4ABC4', zorder=2,label='曝光訊息')
    plt.scatter(data[data["offer_type"] == 'discount'][feature1], data[data["offer_type"] == 'discount'][feature2],c='#766161', zorder=2,label='折扣活動')
    plt.scatter(data[data["offer_type"] == 'bogo'][feature1], data[data["offer_type"] == 'bogo'][feature2],c='#7868E6', zorder=2,label='促銷活動')


#繪製散佈圖中三個類型行銷類別的座標標記
#參數:數據來源表,篩選欄位一,篩選欄位二
def scatterText(data,feature1,feature2):
    for i, row in data[data["offer_type"] == 'informational'].iterrows():
        plt.text(row[feature1], row[feature2]+0.3, f"({row[feature1]}, {row[feature2]})", ha='center', va='bottom', fontsize=10)
    for i, row in data[data["offer_type"] == 'discount'].iterrows():
        plt.text(row[feature1], row[feature2]+0.3, f"({row[feature1]}, {row[feature2]})", ha='center', va='bottom', fontsize=10)
    for i, row in data[data["offer_type"] == 'bogo'].iterrows():
        plt.text(row[feature1], row[feature2]+0.3, f"({row[feature1]}, {row[feature2]})", ha='center', va='bottom', fontsize=10)



#將pct的數值*總計數量 及為該類別的實際數值
#參數:數據來源表,pct在pie圖中會自動生成寫pct即可
def percent_with_actualcount(data,pct):
    total = sum(data)
    absolute = round((pct / 100 * total),0)
    return f'{pct:.2f}% '+f'({absolute:.0f})'

def text_setting():
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    font_path = 'C:\\Windows\\Fonts\\msjhl.ttc'
    font_prop = fm.FontProperties(fname=font_path)
    font_prop.set_style('normal')
    font_prop.set_size('20')
       
        
#計算第1~30天走期中每天有多少筆指定觀測的offer階段
#參數:開始日,結束日+1,數據來源表,欲計算的欄位,指定的offer階段    
def count_interval(x_start,x_end,raw_df,column_selected,offer_type):
    result_np = np.array([0 for i in range(x_start,x_end)])
    for rounds in range(x_start,x_end) :
        filter_by_days = raw_df[raw_df[column_selected] == rounds ] 
        count_informational = (filter_by_days['offer_type'] == offer_type).sum()
        result_np[rounds - 1] = count_informational
        rounds +=1
    return result_np


#繪製offer_recevied和 offer_viewd的折線圖
#參數:訊息接收數量,有效點閱數量
def offerPlot(received_count,effective_viewd_count):
    time = np.array([i for i in range(1,31)])
    plt.figure(figsize=(20,12))
    plt.plot(time,received_count,color ='#FF8080',label='訊息接收',linewidth=2)
    plt.plot(time,effective_viewd_count,color='#5C2FC2',label='訊息點閱',linewidth=2)
    y1 = 0
    y2 = 0
    #繪製圖表標記文字，如果標記為0則不呈現
    for x, y in zip(time, received_count):
        if y != 0:
            plt.text(x, y + 100, f'{y}', ha='center', va='bottom', fontsize=14, color='#FF8080')
            y1 += y
    for x, y in zip(time, effective_viewd_count):
        if y != 0:
            plt.text(x, y + 100, f'{y}', ha='center', va='bottom', fontsize=14, color='#5C2FC2')
            y2 += y
    print("y1=",y1," y2=",y2)

#漏斗圖繪製
def funnel(record_info,effective_offer_transaction,topic,topic_name):
    labels = ["點數接收", "訊息點閱", "訊息接收"]
    #篩選某個主題的有效點閱
    mask_effective_view = (record_info['offer_type'] == topic) & (record_info['effective_viewed_time(day)'] >= 0)
    #篩選某個主題的有效購買
    mask_effective_complete = (record_info['offer_type'] == topic) & (record_info['effective_completed_time(day)'] >= 0)
    #有效點閱的筆數
    effective_view_count =record_info[mask_effective_view].shape[0]
    #有效購買的筆數
    effective_complete_count = record_info[mask_effective_complete].shape[0]
    #將指定主題的相關人數放到list[發送,有效觀看,有效購買]
    person_amount =[(record_info['offer_type'] == topic).sum(),effective_view_count,effective_complete_count]
    #指定主題有效購買的總數
    transaction_amount = sum(effective_offer_transaction[effective_offer_transaction['offer_type'] == topic]['amount'])
    #平均每人有效購買金額
    avg_transaction_amount = transaction_amount / effective_complete_count
    #有效購買發出去的點數
    transaction_reward_amount = sum(effective_offer_transaction[effective_offer_transaction['offer_type'] == topic]['reward'])

    #有效觀看筆例
    view_percent = effective_view_count / person_amount[0] *100
    #平均有效點閱時間
    avg_view_hour = sum((record_info[mask_effective_view]['viewed_time']) -  (record_info[mask_effective_view]['received_time'])) /effective_view_count 
    #有效觀看在時限內的平均落點
    avg_view_duration_percent = sum(((record_info[mask_effective_view]['viewed_time']) -  (record_info[mask_effective_view]['received_time'])) / (record_info[mask_effective_view]['duration']*24)) /effective_view_count *100
    
    #有效購買佔總體發送的筆例
    complete_percent = effective_complete_count / person_amount[0] *100
    #平均有效購買時間
    avg_transaction_hour= sum((record_info[mask_effective_complete]['completed_time']) - (record_info[mask_effective_complete]['received_time'])) / effective_complete_count
    #有購買看在時限內的平均落點
    avg_completed_duration_percent=  sum(((record_info[mask_effective_complete]['completed_time']) -  (record_info[mask_effective_complete]['received_time'])) / (record_info[mask_effective_complete]['duration']*24)) /effective_complete_count *100
    
    #繪製每層漏斗圖
    colors = ["#EFB495",'#F7A4A4',"#ADA2FF"]
    fig = plt.figure(figsize=(12,10))
    plt.fill_betweenx(y=[1.8, 5], x1=[9,13], x2=[9,5], color=colors[0], edgecolor="#aeb6bf", linewidth=3);
    plt.fill_betweenx(y=[5.1, 7], x1=[13,15], x2=[5,3], color=colors[1], edgecolor="#aeb6bf", linewidth=3);
    plt.fill_betweenx(y=[7.1, 8.5], x1=[15,16], x2=[3,2], color=colors[2], edgecolor="#aeb6bf", linewidth=3);


    plt.xticks([],[]);
    plt.yticks([3.7,6.1,7.7], labels,fontsize = 18);
    plt.text(9, 7.6, f'人次:{person_amount[0]}', fontsize=20, color="black", ha="center");
    plt.text(9, 5.6, f'點閱率:{view_percent:.2f}% ({person_amount[1]}人次) \n 平均點閱:{avg_view_hour:.2f}/小時 \n 平均時限(%):{avg_view_duration_percent:.2f}%' , fontsize=20, color="black", ha="center");
    plt.text(9, 3, f'轉換率:{complete_percent:.2f}% ({person_amount[2]}人次)\n 平均交易:{avg_transaction_hour:.2f}/小時 \n 平均時限(%):{avg_completed_duration_percent:.2f}% \n 總金額:{transaction_amount:.2f} \n 平均客單:{avg_transaction_amount:.2f} \n 發放點數:{transaction_reward_amount}', fontsize=20, color="black", ha="center");
    plt.title(topic_name +"活動漏斗", loc="center", fontsize=25);




#標籤重新分類，讓購買傾向分級
def calculate_label(labels):
    labels = labels.values   
    if 2 in labels:
       if all(l == 2 for l in labels):
           return 20 #每次都買
       else:
           return 21 #有買過非每次買
    elif 1 in labels: 
       if all(l == 1 for l in labels):
           return 10 #每次都看但不買
       else:
           return 11 #不買也不是每次都看
    elif all(l == 0 for l in labels):
       return 0   #從來沒看過
    else:
       return -1  # 如果沒有符合條件，設置為 -1
   

#給予member 新label並返回會員基礎資料和label欄位，其餘欄位刪除
def process_group(df):
    # 根據 member_id 分組，並對每個分組應用 calculate_label 函數
    result = df.groupby('member_id').apply(lambda x: x.assign(label=calculate_label(x['label'])))
    # 刪除多餘欄位，只保留指定欄位
    return result[['member_id', 'label', 'gender', 'age', 'years_till_now', 'income']].drop_duplicates()


#繪製同性別同年齡於社群管道購買傾向圖(%)、同性別同年齡於社群管道購買傾向人數
#參數:年齡分割集,會員唯一值數據,行銷類別
def purchase_percentage(bins_age,unique_member_newcomer,topic_name):
    #年齡分組(包下不包上)
    age_labels = bins_age[:-1]
    unique_member_newcomer['age_group'] = pd.cut(unique_member_newcomer['age'], bins=bins_age, labels= age_labels, right=False)
    
    #建立27*4 pd (年齡9組，性別3組) ，用於放入購買傾向(%)、購買人數
    age_gender_label_percent = pd.DataFrame(np.zeros((27,4)),columns=['age','gender','purchase_intention','person_amout'])
    age_gender_label_percent['age'] = sorted(bins_age[:-1] * 3 )
    gender_list = list(unique_member_newcomer['gender'].unique())
    age_gender_label_percent['gender'] = (gender_list * 9)

    #將標籤為20|21在同年齡同性別的購買傾向(%)
    for i in range(age_gender_label_percent.shape[0]):
        mask_gender_age = (unique_member_newcomer['age_group'] == age_gender_label_percent.loc[i,'age']) & (unique_member_newcomer['gender'] == age_gender_label_percent.loc[i,'gender'])
        mask_gender_age_label =mask_gender_age & ((unique_member_newcomer['label'] ==20) | (unique_member_newcomer['label'] ==21))
        #如果購買人數為0則購買傾向(%)、購買人數為0
        if  unique_member_newcomer[mask_gender_age].shape[0] == 0:
            age_gender_label_percent.loc[i,'purchase_intention'] = 0
            age_gender_label_percent.loc[i,'person_amout'] = 0
        else:
            purchse_intention = unique_member_newcomer[mask_gender_age_label].shape[0] / unique_member_newcomer[mask_gender_age].shape[0]
            age_gender_label_percent.loc[i,'purchase_intention']=purchse_intention
            age_gender_label_percent.loc[i,'person_amout'] = unique_member_newcomer[mask_gender_age_label].shape[0]

    #圖表的Y值
    age_x_axis = bins_age[:-1]
    female_label =age_gender_label_percent[age_gender_label_percent['gender'] == 'F']['purchase_intention']
    male_label = age_gender_label_percent[age_gender_label_percent['gender'] == 'M']['purchase_intention']
    other_label = age_gender_label_percent[age_gender_label_percent['gender'] == 'O']['purchase_intention']

    plt.figure(figsize=(10,8))
    X_axis = np.arange(len(age_x_axis)) 
    #一組有3個bar 每組中bar的位置需要左右位移才不會重疊
    plt.bar(X_axis - 0.25, female_label, 0.25, label = '女', color ='#DC8686' ,zorder=2 ) 
    plt.bar(X_axis, male_label, 0.25, label = '男', color ='#BAABDA',zorder=2) 
    plt.bar(X_axis + 0.25, other_label, 0.25, label = '其他', color ='#A5B68D',zorder=2)
    
    #顯示圖表標記文字
    index = 0
    for j in female_label :
        plt.text(index -0.25, j+0.01,f'{j*100:.0f}%', ha='center', va='bottom', fontsize=10)
        index += 1
      
    index = 0
    for j in male_label :
         plt.text(index, j+0.01,f'{j*100:.0f}%', ha='center', va='bottom', fontsize=10)
         index += 1   

    index = 0
    for j in other_label :
        if j == 0:
            pass
        else:
            plt.text(index +0.25, j+0.01,f'{j*100:.0f}%', ha='center', va='bottom', fontsize=10)
        index += 1

    yticks =np.arange(0,1.1,0.1 )
    #將y軸由小數點轉換為百分筆
    plt.xticks(X_axis, [f'{bins_age[i]}-{bins_age[i+1]}歲' for i in range(len(bins_age)-1)]) 
    plt.yticks(yticks) 
    plt.ylim(0,1.05)
    plt.grid(color = '#D2E0FB', zorder=1)
    plt.xlabel("年齡區間",size =14) 
    plt.ylabel("購買傾向(%)",rotation = 0, ha='right',size =14) 
    plt.title(topic_name+"活動-同年齡區間同性別中購買傾向比例",size =14) 
    plt.legend() 
    plt.show() 
    
    #圖表的Y值
    age_x_axis = bins_age[:-1]
    female_label =age_gender_label_percent[age_gender_label_percent['gender'] == 'F']['person_amout']
    male_label = age_gender_label_percent[age_gender_label_percent['gender'] == 'M']['person_amout']
    other_label = age_gender_label_percent[age_gender_label_percent['gender'] == 'O']['person_amout']

    plt.figure(figsize=(10,8))
    X_axis = np.arange(len(age_x_axis)) 
    
    #將標籤為20|21在同年齡同性別的購買人數(%)
    plt.bar(X_axis - 0.25, female_label, 0.25, label = '女', color ='#DC8686' ,zorder=2 ) 
    plt.bar(X_axis, male_label, 0.25, label = '男', color ='#BAABDA',zorder=2) 
    plt.bar(X_axis + 0.25, other_label, 0.25, label = '其他', color ='#A5B68D',zorder=2)
   
    #顯示圖表標記文字
    index = 0
    for j in female_label :
        plt.text(index -0.25, j+0.01,f'{j:.0f}', ha='center', va='bottom', fontsize=10)
        index += 1
      
    index = 0
    for j in male_label :
         plt.text(index, j+0.01,f'{j:.0f}', ha='center', va='bottom', fontsize=10)
         index += 1   

    index = 0
    for j in other_label :
        if j == 0:
            pass
        else:
            plt.text(index +0.25, j+0.01,f'{j:.0f}', ha='center', va='bottom', fontsize=10)
        index += 1
        
    yticks = np.arange(0,500,100)
    plt.xticks(X_axis, [f'{bins_age[i]}-{bins_age[i+1]}歲' for i in range(len(bins_age)-1)])
    plt.yticks(yticks) 
    plt.ylim(0,500) 
    plt.grid(color = '#D2E0FB', zorder=1)
    plt.xlabel("年齡區間",size =14) 
    plt.ylabel("購買人數",rotation = 0, ha='right',size =14) 
    plt.title(topic_name+"活動-同年齡區間同性別中購買人數",size =14) 
    plt.legend() 
    plt.show()

#決策數計算
def decisionTree(X,y):
    #建立訓練集正確率、測試集正確率、分層數的容器
    max_tree_accuracy_train = pd.Series(np.zeros(86))
    max_tree_accuracy_test = pd.Series(np.zeros(86))
    max_tree_accuracy_depth = pd.Series(np.zeros(86))
    percent = 0.1
    for i in range(86):
        #設定test_size，在同個test_size下，建立訓練集正確率、測試集正確率容器
        XTrain, XTest, yTrain, yTest = tts(X, y, test_size=percent, random_state=1)
        tree_accuracy_train = pd.Series(np.zeros(10))
        tree_accuracy_test = pd.Series(np.zeros(10))
        #execute_time= pd.Series(np.zeros(8))
        #在同個test_size下計算分層數為1~9的正確率與執行時間
        for j in range(1,10):
            #start_time = time.time()
            dt = tree.DecisionTreeClassifier(max_depth=j)
            dt.fit(XTrain, yTrain)
            tree_accuracy_train[j-1] = dt.score(XTrain, yTrain)
            tree_accuracy_test[j-1] = dt.score(XTest, yTest)
            #end_time = time.time()
            '''
            #運行>3秒鐘才須將時間考慮 =>都在3秒內
            if end_time - start_time >1:
                print('test_size:',percent, j,"層",end_time - start_time,"秒")
            '''
        #取訓練與測試正確率差異的絕對值
        percent_diff = abs(tree_accuracy_test - tree_accuracy_train)
        #取測試正確率最高者並確認與訓練正確率的差額絕對值是否小於等於10%
        keepRun = True
        while keepRun:
            max_index=tree_accuracy_test.idxmax()
            #<=10%則該測試正確率對應的分層數及訓練正確率作為該test_size下的最佳解
            if percent_diff[max_index] <= 0.1:
                max_tree_accuracy_train[i] = tree_accuracy_train[max_index]
                max_tree_accuracy_test[i] = tree_accuracy_test[max_index]
                max_tree_accuracy_depth[i] = max_index + 1
                keepRun = False
            #如果>10%則將該測試正確率改為0，確保循環取最大值時不會被取到
            else:
                tree_accuracy_test[max_index] = 0
        percent += 0.01
    return  max_tree_accuracy_train,max_tree_accuracy_test,max_tree_accuracy_depth
    

#鄰近演算
def neighbor(X,y):
    #建立訓練集正確率、測試集正確率、分層數的容器
    max_accuracy_train = pd.Series(np.zeros(36))
    max_accuracy_test = pd.Series(np.zeros(36))
    accuracy_n_neighbors = pd.Series(np.zeros(36))
    percent = 0.6
    
    #設定test_size，在同個test_size下，建立訓練集正確率、測試集正確率容器
    for i in range(36):
        XTrain, XTest, yTrain, yTest = tts(X, y, test_size=percent, random_state=1)
        accuracy_train = pd.Series(np.zeros(49))
        accuracy_test = pd.Series(np.zeros(49))
        #execute_time= pd.Series(np.zeros(60))
  
        #在同個test_size下計算k值為4~52的正確率與執行時間
        for j in range(4,53): #8~50
            start_time = time.time()
            knn = neighbors.KNeighborsClassifier(n_neighbors=j)
            knn.fit(XTrain, yTrain)
            accuracy_train[j-4] = knn.score(XTrain, yTrain)
            accuracy_test[j-4] = knn.score(XTest, yTest)
            end_time = time.time()
       
            if end_time - start_time >1:
                print('test_size:',percent, j,"層",end_time - start_time,"秒")
       #取訓練與測試正確率差異的絕對值
        percent_diff = abs(accuracy_train - accuracy_test)
        keepRun = True
        #取測試正確率最高者並確認與訓練正確率的差額絕對值是否小於等於10%
        
        while keepRun:
            max_index = accuracy_test.idxmax()
            #<=10%則該測試正確率對應的分層數及訓練正確率作為該test_size下的最佳解
            if percent_diff[max_index] <= 0.1:
                max_accuracy_train[i] = accuracy_train[max_index]
                max_accuracy_test[i] = accuracy_test[max_index]
                accuracy_n_neighbors[i] = max_index + 4
                keepRun = False
            #如果>10%則將該測試正確率改為0，確保循環取最大值時不會被取到
            else:
                accuracy_test[max_index] = 0
        percent += 0.01
    return max_accuracy_train,max_accuracy_test,accuracy_n_neighbors


#繪製散佈圖矩陣
def create_pairplot(parallel_df,labelNumber,labelColor,titleCustomer):
    #製圖
    pairplot = sns.pairplot(parallel_df[parallel_df['標籤'] ==labelNumber], hue="標籤",palette={labelNumber: labelColor})
    #設定title位置
    pairplot.fig.suptitle('散佈圖矩陣'+titleCustomer,y=1.03,fontsize=14)
    #設定y軸水平靠右
    for ax in pairplot.axes.flat:
        ax.set_ylabel(ax.get_ylabel(),fontsize=12, ha='right',rotation=0)  # Y 軸標籤水平靠右
    print('標籤:',labelNumber,'筆數:',(parallel_df['標籤'] ==labelNumber).sum())
    

    