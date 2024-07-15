from rouge_score import rouge_scorer

# 文章AとBを定義
article_A = "The quick brown fox jumps over the lazy dog. The dog, seemingly unperturbed, yawns and continues to lie there."
summary_B = "The quick brown fox jumps over the lazy dog, who remains unperturbed."

# ROUGEスコアを計算
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(article_A, summary_B)

# 結果を表示
print("ROUGE-1スコア:", scores['rouge1'])
print("ROUGE-Lスコア:", scores['rougeL'])


# これを日本語化したい