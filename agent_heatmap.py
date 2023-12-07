import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_sentiment_heatmap(df,agent,path):
    
    # Assuming your DataFrame has a column named 'TopSentimentScores'


    if type(df['TopSentimentScores'][0]) is str:
        df['TopSentimentScores']=df['TopSentimentScores'].apply(eval)
    # Create a new DataFrame with columns 'Index', 'Topic', 'SentimentType', and 'SentimentScore'
    df=df[df['Character']==agent][['TopSentimentScores']]
    data = []
    for index, row in df.iterrows():
        for topic, (sentiment_type, sentiment_score) in row['TopSentimentScores'].items():
            data.append([index, topic, sentiment_type, sentiment_score])

    columns = ['Index', 'Topic', 'SentimentType', 'SentimentScore']
    new_df = pd.DataFrame(data, columns=columns)
    neg_cond=new_df['SentimentType']=="Negative"
    pos_cond=new_df['SentimentType']=="Positive"
    new_df.loc[neg_cond,'SentimentScore']=new_df.loc[neg_cond,'SentimentScore']*-1

    new_df.loc[pos_cond,'SentimentScore']=new_df.loc[pos_cond,'SentimentScore']+1

    #Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = new_df.pivot(index='Topic', columns='Index', values='SentimentScore')

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data,
                cmap="coolwarm", annot=True,linewidths=0.01,annot_kws={'rotation': 90})
    plt.title(agent+'Sentiment Heatmap')
    plt.savefig(path)
    plt.show()
    plt.close()

# Example usage with your DataFrame
# create_sentiment_heatmap(your_dataframe)
