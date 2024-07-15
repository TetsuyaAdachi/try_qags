from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import QuestionAnsweringPipeline

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from get_rouge_score import get_rouge_score
from transformers import pipeline


# 参考
# https://zenn.dev/ty_nlp/articles/aaad1aec70d53e

INDEX = 1

article_path = f"docs/doc_{INDEX}/document.txt"
summary_path = f"docs/doc_{INDEX}/gpt_result.txt"

with open(article_path) as fa:
    original_text = fa.read()
with open(summary_path) as fs:
    summary_text = fs.read()

# model_name = 'KoichiYasuoka/bert-base-japanese-wikipedia-ud-head'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# qap = QuestionAnsweringPipeline(tokenizer=tokenizer, model=model, align_to_words=False)


model_name = "tsmatz/roberta_qa_japanese"
qap = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=model_name)



def score_summaries(original_text, summary_text):
    # 要約対象の文書と作成した要約

    # 質問を生成します。
    questions = [
        "議事録ではどのような話題が議論されましたか？",
        "議事録で取り上げられた主要な意見や提案は何でしたか？",
        "議事録に記録されているアクションアイテムや決定事項は何ですか？",
        "議事録で特に強調されているポイントや重要な情報は何ですか？",
        "議事録の要約において、元の文書の主要な内容や議論の核心が適切に反映されていましたか？"
    ]
    # questions = generate_question(original_text)
    # 文書と要約それぞれから回答を得ます。
    scores = []
    for question in questions:
        answer_from_doc = qap(question=question, context=original_text)
        answer_from_summary = qap(question=question, context=summary_text)
        print(f"Question: {question}")
        print(f"Answer from Document: {answer_from_doc['answer']}")
        print(f"Answer from Summary: {answer_from_summary['answer']}")

        scores.append(get_rouge_score(answer_from_doc['answer'], answer_from_summary['answer']))

    return sum(scores) / len(questions)
    # # 文書と要約それぞれから回答を得ます。
    # answers_from_doc = [question_answering(question=q['question'], context=original_text) for q in questions]
    # answers_from_summary = [question_answering(question=q['question'], context=summary_text) for q in questions]
    # print(answers_from_doc)
    # print(answers_from_summary)



if __name__ == "__main__":
    # 点数を計算
    score = score_summaries(original_text, summary_text)
    print(f"Summary Score: {score}")
