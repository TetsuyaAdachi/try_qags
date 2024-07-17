import MeCab


def safe_get(lst, index, default=None):
    try:
        return lst[index]
    except IndexError:
        return default


def get_keyword(text: str):
    # 形態素解析器のインスタンス生成

    # NEologd辞書のパスを指定
    dicdir = "/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd"  # インストール時に表示されたパスに置き換えてください
    mecab = MeCab.Tagger(f"-d {dicdir}")

    # 解析結果を取得
    node = mecab.parseToNode(text)

    # 固有名詞を抽出
    results = []
    while node:
        features = node.feature.split(",")
        if features[0] == "名詞" and features[1] == "固有名詞":
            results.append(node.surface)
        node = node.next

    return list(set(results))
