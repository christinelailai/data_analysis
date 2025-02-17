import pymysql
try:
    connect = pymysql.connect(host='127.0.0.1',
                    user='root',
                    password = "password",
                    port = 3306
        )
        
    sql_1='''
    CREATE TABLE IF NOT EXISTS project_app_data.member_info
    (id VARCHAR(50) PRIMARY KEY,gender VARCHAR(10),age INT,became_member_date DATE,income INT);
    '''
    sql_2='''
    CREATE TABLE IF NOT EXISTS project_app_data.offer_info
    (id VARCHAR(50) PRIMARY KEY,reward INT,channels VARCHAR(80),difficulty INT,duration INT,offer_type VARCHAR(40));  
    '''  
    
    #因為member_id offer_id completed_time會有重複(同一方案不同時間點發送，但在同一筆交易中完成兩個offer)，故此table先不設定primary key
    sql_3 =''' 
    CREATE TABLE IF NOT EXISTS project_app_data.offer_record
    (member_id VARCHAR(50),offer_id VARCHAR(50),received_time INT,viewed_time INT,completed_time INT,PRIMARY);
    '''
    sql_4 = '''
    CREATE TABLE IF NOT EXISTS project_app_data.transaction_record
     (member_id VARCHAR(50),amount INT,transaction_time INT,PRIMARY KEY(member_id,transaction_time));
     '''
  
    cursor = connect.cursor() 
    for i in [sql_1,sql_2,sql_3,sql_4]:
        cursor.execute(i)
    connect.commit()
    connect.close()
    

except Exception as e:
    print(e)