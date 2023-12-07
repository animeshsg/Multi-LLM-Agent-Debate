from bert_score import BERTScorer
import pandas as pd
import numpy as np

def get_bertscore(candidate,reference):
    '''
    Input: Candidate and Reference text
    Output : Precision , Recall and F1 Score of BERTScore 
    '''
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([candidate], [reference])
    P=float(P)
    R=float(R)
    F1=float(F1)
    return P,R,F1

def get_bertscore_fit(df_result):
    df_agent1=df_result[df_result['Character'].eq("Agent1")][['F1_Score_Seed']]
    df_agent2=df_result[df_result['Character'].eq("Agent2")][['F1_Score_Seed']]
    y1=df_agent1.F1_Score_Seed.values.tolist()[1:]
    y2=df_agent2.F1_Score_Seed.values.tolist()[1:]
    time1=np.array([i for i in range(len(y1))])
    time2=np.array([i for i in range(len(y2))])
    coefficients1 = np.polyfit(time1, y1, 1)
    coefficients2 = np.polyfit(time2, y2, 1)

    # Get the slope and intercept from the coefficients
    slope1 = coefficients1[0]
    slope2 = coefficients2[0]
    return slope1, slope2

def get_bertscore_summary(df_result):
    agent1_mean=df_result[df_result['Character'].eq("Agent1")].F1_Score_Seed.mean()
    agent2_mean=df_result[df_result['Character'].eq("Agent2")].F1_Score_Seed.mean()
    agent1_mean_perplexity=df_result[df_result['Character'].eq("Agent1")].Perplexity.mean()
    agent2_mean_perplexity=df_result[df_result['Character'].eq("Agent2")].Perplexity.mean()
    return agent1_mean,agent2_mean,agent1_mean_perplexity,agent2_mean_perplexity

def get_sentiment_counts(data):
    # Initialize counts
    if type(data) is str:
        data=eval(data)

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Iterate through the dictionary values
    for sentiment, score in data.values():
        if sentiment == 'Positive':
            positive_count += 1
        elif sentiment == 'Negative':
            negative_count += 1
        elif sentiment == 'Neutral':
            neutral_count += 1

    # Create a 3-length tuple
    sentiment_counts = [positive_count,neutral_count, negative_count]
    
    return sentiment_counts
