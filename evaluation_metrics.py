from TextFileTransformer import TextFileTransformer
import pandas as  pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

for i in range(12):
    file_path = 'KData/response' + str(i) + '.txt' 
    try:
        text_transformer = TextFileTransformer(file_path)

        df_result = text_transformer.transform(None)
        df = pd.DataFrame(data=df_result)
        df.to_csv('KData/eval_metric_100_' + str(i) + '.csv')
        print(file_path+" Processed Successfully")
    except Exception as e:
        print(e)
        print(file_path+" : Couldnt be processed")


