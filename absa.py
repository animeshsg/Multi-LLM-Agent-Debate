from bertopic import BERTopic
import os 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from umap import UMAP
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer

class Absa():
    def __init__(self,data):

        # bertopic Tokenizers and model params and init
        
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.topic_model = BERTopic(
                            n_gram_range=(1, 2),
                            vectorizer_model=self.vectorizer_model,
                            nr_topics='auto',
                            min_topic_size=5,
                            top_n_words=5,
                            calculate_probabilities=True)
        self.model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        
        #ABSA Tokenizers and model params init
        self.absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
        self.absa_tokenizer = AutoTokenizer.from_pretrained(self.absa_model_name)
        self.absa_model = AutoModelForSequenceClassification.from_pretrained(self.absa_model_name)

        #Input
        self.data=data
    
    def get_topics(self):
        convs=self.data.Dialogue.to_list()
        corpus_embeddings = self.model_embedding.encode(convs)
        self.topic_model=self.topic_model.fit(convs,corpus_embeddings)
        topics, probabilities = self.topic_model.transform(convs, corpus_embeddings)
        new_topics = self.topic_model.reduce_outliers(convs, topics, strategy="c-tf-idf")
        self.topic_model.update_topics(convs, topics=new_topics, vectorizer_model=self.vectorizer_model)
        print(self.topic_model.get_topic_freq())
        doc_info=self.topic_model.get_document_info(convs)
        doc_info=doc_info[['Document','Representation']]
        doc_info = doc_info.rename(columns={'Document': 'Dialogue'})
        self.data=pd.merge(self.data,doc_info,on="Dialogue",how="inner")
        return self.data
    
    def get_aspect_scores(self,row):
        #Get sentiment scores given a text and its aspect list
        text=row['Dialogue']
        aspect_list=row['Representation']
        sentiment_aspect = {}
        for aspect in aspect_list:
            inputs = self.absa_tokenizer(text, aspect, return_tensors="pt")

            with torch.inference_mode():
                outputs = self.absa_model(**inputs)

            scores = F.softmax(outputs.logits[0], dim=-1)
            label_id = torch.argmax(scores).item()
            sentiment_aspect[aspect] = (self.absa_model.config.id2label[label_id], scores[label_id].item())

        return sentiment_aspect

    def get_absa(self):
        #  Main Function to get the absa sentiment scores 
        self.data=self.get_topics()
        self.data['sentiment_scores'] = self.data.apply(self.get_aspect_scores, axis=1)
        self.data = self.data.drop(['Representation'], axis=1)
        return self.data