from rouge_score import rouge_scorer
from janome.tokenizer import Tokenizer

from rouge_score.tokenize import SPACES_RE


# 参考
# https://nikkie-ftnext.hatenablog.com/entry/why-rouge-score-library-cannot-calculate-from-japanese-texts

class NonAlphaNumericSupportTokenizer(Tokenizer):
    """
    >>> NonAlphaNumericSupportTokenizer().tokenize("いぬねこ")
    ['いぬ', 'ねこ']
    """
    def tokenize(self, text):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(text)
        wakati =  " ".join([token.surface for token in tokens])
        return SPACES_RE.split(wakati.lower())
    
def get_rouge_score(article: str, summary: str):
    """
    Args:
        article (str): 本文です。
        summary (str): 要約です。

    Returns:
        戻り値のデータ型: 戻り値の説明。

    Examples:
        >>> 結果 = 関数名(引数1の例, 引数2の例)
        >>> print(結果)
        結果の説明
    """
    # 文章AとBを定義
    # article = "素早い茶色の狐がのんびりした犬の上を飛び越えます。犬は気にせずにあくびをして、そこに横たわり続けます。"
    # summary = "素早い茶色の狐がのんびりした犬の上を飛び越え、犬は気にしません。"



    # ROUGEスコアを計算
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], tokenizer=NonAlphaNumericSupportTokenizer())  
    scores = scorer.score(article, summary)

    # 結果を表示
    print("ROUGE-1スコア:", scores['rouge1'])
    print("ROUGE-Lスコア:", scores['rougeL'])

if __name__ == "__main__":

    INDEX = 10

    article_path = f"docs/doc_{INDEX}/document.txt"
    summary_path = f"docs/doc_{INDEX}/gpt_result.txt"

    with open(article_path) as fa:
        article = fa.read()
    with open(summary_path) as fs:
        summary = fs.read()
    
    get_rouge_score(article, summary)