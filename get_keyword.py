import MeCab


def safe_get(lst, index, default=None):
    try:
        return lst[index]
    except IndexError:
        return default


def get_keyword(text: str):
    # 形態素解析器のインスタンス生成
    m = MeCab.Tagger()

    # テキストを形態素解析
    node = m.parse(text)

    # 固有名詞を抽出
    keywords = []
    for line in node.split("\n"):
        word_class = safe_get(line.split("\t"), 4)
        if word_class and word_class == "名詞-普通名詞-一般":
            keywords.append(line.split("\t")[0])

    result = list(set(keywords))
    return result
