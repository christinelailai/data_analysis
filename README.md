一、資料來源
https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data


二、來源資料檔
API(1個)
https://storage.googleapis.com/kagglesdsdata/datasets/690741/1210253/portfolio.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241117%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241117T130030Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7decb9492ff721bb2db3c2599da0f4076193cafeb810490c90cf85895e6098699810e92a895b0ac56a43504daa37adaa3fc4495ff9864fc8f9a01d00afe2911d66a8ffba378886e5adabfe4b064d5a031e060a54a94076b4f4dd9b32aa9b73291316b5f13f9f6713ce403a35388e3240222e27ff250e0881e88b525c913ce4f273939fd6320093babf1181ac1d545ad24dda6fe24e0176a461f70faa16b4c9953effc0f8e348516d162003c0b4d10e766e265113309512ded4448b031f8c0b3abfb0e62c4d2a9a7bf0579d0ac4b183e7e8dd54d3d037a6932a4bfc2b9d489b848f5a1d1e27ef1405d3d75c2d702171009d02b22d63168855fcb1657531f14b3d
json檔案(2個):見檔案資料夾

三、python
create_SQLTable:創建mysql資料表
preprocess_insertSQL:將資料載入、清洗、放入DB
plot_des:既有客戶樣貌分析
self_def:所有自訂函數
campaign analysis:類別相關圖表與計算(依據方案的客戶行為百分比堆疊長條圖在member_analysis)
member_analysis:依據方案的客戶行為百分比堆疊長條圖、客戶行為預測的相關圖表

四、其他說明
有寫些地方有用pd.merge，如果出現key error 就是有重複運行pd.merge造成欄位重複，pd會自動把重複欄位變成原欄位名_X 原欄位名_Y，故出現key error表示欄位已經壞掉了
請重跑整份檔案比較快
