from rouge_score import rouge_scorer
from janome.tokenizer import Tokenizer

from rouge_score.tokenize import SPACES_RE

class NonAlphaNumericSupportTokenizer(Tokenizer):
    def tokenize(self, text):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(text)
        wakati =  " ".join([token.surface for token in tokens])
        return SPACES_RE.split(wakati.lower())
    

def get_rouge_score(article: str, summary: str):

    # ROUGEスコアを計算
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], tokenizer=NonAlphaNumericSupportTokenizer())  
    scores = scorer.score(article, summary)

    # 結果を表示
    return scores['rouge1'].fmeasure
