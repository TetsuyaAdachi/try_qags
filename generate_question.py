def generate_question(summary_text, answers):
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    model_name_or_path = "sonoisa/t5-base-japanese-question-generation"

    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, is_fast=True)

    from tqdm.auto import tqdm

    INPUT_MAX_LEN = 512  # モデルに入力されるトークン列の最大長。最大長を超えたトークンは切り捨てられる。
    OUTPUT_MAX_LEN = 64  # モデルから出力されるトークン列の最大長。最大長を超えないように文が生成されるはず。

    answer_context_list = []

    context = summary_text
    for answer in answers:
        answer_context_list.append((answer, context))

    generated_questions = []

    for answer, context in tqdm(answer_context_list):
        # モデルに入力可能な形式に変換する。
        input = f"answer: {answer} context: {context}"

        # 入力文をトークナイズする。
        tokenized_inputs = tokenizer.batch_encode_plus(
            [input], max_length=INPUT_MAX_LEN, truncation=True, 
            padding="longest", return_tensors="pt")

        input_ids = tokenized_inputs['input_ids']
        input_mask = tokenized_inputs['attention_mask']
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        # 問題文を生成する。
        tokenized_outputs = model.generate(input_ids=input_ids, attention_mask=input_mask, 
            max_length=OUTPUT_MAX_LEN, return_dict_in_generate=True, decoder_start_token_id=0,
            temperature=0.0,  # 生成にランダム性を入れる温度パラメータ
            num_beams=4,  # ビームサーチの探索幅
            # diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
            # num_beam_groups=4,  # ビームサーチのグループ
            num_return_sequences=1,  # 生成する文の数
        )

        # 生成された問題文のトークン列を文字列に変換する。
        outputs = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
            for ids in tokenized_outputs.sequences]

        generated_questions.append(outputs) 

    return generated_questions