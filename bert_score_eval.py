from bert_score import BERTScorer

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