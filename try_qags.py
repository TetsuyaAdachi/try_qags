from transformers import pipeline

from get_keyword import get_keyword
from get_rouge_score import get_rouge_score
from generate_question import generate_question
import os
# 参考
# https://zenn.dev/ty_nlp/articles/aaad1aec70d53e


model_name = "tsmatz/roberta_qa_japanese"
qap = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=model_name,
    device=0,
)


def score_summaries(original_text, summary_text, answers):
    # 要約対象の文書と作成した要約

    questions = generate_question(summary_text, answers)

    # 文書と要約それぞれから回答を得ます。
    scores = []
    answer_documents = []
    answer_summaries = []
    for question in questions:
        answer_from_doc = qap(
            question=question,
            context=original_text,
            align_to_words=False,
        )
        answer_from_summary = qap(
            question=question,
            context=summary_text,
            align_to_words=False,
        )
        answer_documents.append(answer_from_doc["answer"])
        answer_summaries.append(answer_from_summary["answer"])

        score = get_rouge_score(
            answer_from_doc["answer"], answer_from_summary["answer"]
        )
        scores.append(score)

    avg = sum(scores) / len(questions)
    return (
        round(avg, 3),
        questions,
        answer_documents,
        answer_summaries,
    )


if __name__ == "__main__":
    MAX_NUM = 30

    for index in range(1, MAX_NUM):
        article_path = f"docs/doc_{index}/document.txt"
        summary_path = f"docs/doc_{index}/gpt_result.txt"
        annotation_path = f"docs/doc_{index}/annotation.csv"

        with open(article_path) as fa:
            original_text = fa.read()
        with open(summary_path) as fs:
            summary_text = fs.read()
        # with open(annotation_path) as fb:
        #     csvreader = csv.reader(fb)
        #     answers = next(csvreader)
        answers = get_keyword(summary_text)

        score, questions, answer_documents, answer_summaries = score_summaries(
            original_text, summary_text, answers
        )
        print(f"Index: {index},  Summary Score: {score}")

        result_score_path = f"results/doc_{index}/score.txt"
        question_path = f"results/doc_{index}/question.txt"
        answer_document_path = f"results/doc_{index}/answer_document.txt"
        answer_summery_path = f"results/doc_{index}/answer_summery.txt"
        key_word_path = f"results/doc_{index}/key_word.txt"

        # ファイルのディレクトリが存在しない場合、作成する
        os.makedirs(os.path.dirname(result_score_path), exist_ok=True)
        os.makedirs(os.path.dirname(question_path), exist_ok=True)
        os.makedirs(os.path.dirname(answer_document_path), exist_ok=True)
        os.makedirs(os.path.dirname(answer_summery_path), exist_ok=True)
        os.makedirs(os.path.dirname(key_word_path), exist_ok=True)

        with open(result_score_path, "w") as file:
            file.write(str(score))

        with open(question_path, "w") as file:
            for item in questions:
                file.write(item[0] + "\n\n")

        with open(answer_document_path, "w") as file:
            for item in answer_documents:
                file.write(item + "\n\n")

        with open(answer_summery_path, "w") as file:
            for item in answer_summaries:
                file.write(item + "\n\n")

        with open(key_word_path, "w") as file:
            for item in answers:
                file.write(item + "\n\n")
