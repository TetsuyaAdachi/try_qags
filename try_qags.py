from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import QuestionAnsweringPipeline

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from get_rouge_score import get_rouge_score
from transformers import pipeline
import csv
from generate_question import generate_question

# 参考
# https://zenn.dev/ty_nlp/articles/aaad1aec70d53e

INDEX = 1

article_path = f"docs/doc_{INDEX}/document.txt"
summary_path = f"docs/doc_{INDEX}/gpt_result.txt"
annotation_path = f"docs/doc_{INDEX}/annotation.csv"

with open(article_path) as fa:
    original_text = fa.read()
with open(summary_path) as fs:
    summary_text = fs.read()
with open(annotation_path) as fb:
    csvreader = csv.reader(fb)
    answers = next(csvreader)

model_name = "tsmatz/roberta_qa_japanese"
qap = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=model_name)


def score_summaries(original_text, summary_text):
    # 要約対象の文書と作成した要約

    questions = generate_question(summary_text, answers)

    # 文書と要約それぞれから回答を得ます。
    scores = []
    for question in questions:
        answer_from_doc = qap(question=question, context=original_text)
        answer_from_summary = qap(question=question, context=summary_text)
        print(f"Question: {question}")
        print(f"Answer from Document: {answer_from_doc['answer']}")
        print(f"Answer from Summary: {answer_from_summary['answer']}")
        score = get_rouge_score(answer_from_doc['answer'], answer_from_summary['answer'])
        print(score)
        scores.append(score)

    return sum(scores) / len(questions)



if __name__ == "__main__":
    # 点数を計算
    score = score_summaries(original_text, summary_text)
    print(f"Summary Score: {score}")
