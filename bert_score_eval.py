from bert_score import BERTScorer

def get_bertscore(candidate,reference):
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([candidate], [reference])
    P=float(P)
    R=float(R)
    F1=float(F1)
    return P,R,F1