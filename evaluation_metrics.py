from TextFileTransformer import TextFileTransformer
import pandas as  pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

for i in range(12):
    file_path = 'response_20_' + str(i) + '.txt' 
    text_transformer = TextFileTransformer(file_path)

    df_result = text_transformer.transform(None)
    df = pd.DataFrame(data=df_result)
    df.to_csv('eval_metric' + str(i) + '.csv')


