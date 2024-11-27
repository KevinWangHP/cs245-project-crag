import re
import random


def add_noise(text, rng=random):
    # 定义可用的噪声函数
    noise_functions = [add_spaces_at_start_end]
    
    # 纠错、改错类需求，不能随意破坏
    if not any([
        kw in text.lower() for kw in ('纠错', '改错', '纠正', '改正', 'correct')
    ]) and len(text) > 2:
        noise_functions.extend([
            add_spaces_inside,
            swap_adjacent_characters,
            repeat_characters,
            change_punctuation_width
        ])
    
    rng.shuffle(noise_functions)
    for func in noise_functions:
        text, nosiy = func(text, rng)
        if nosiy:
            break
    return text


def add_spaces_at_start_end(text, rng=random):
    """在开头结尾添加空格、\n 和 \t"""
    nosiy = False
    for _ in range(rng.randint(1, 4)):
        space = rng.choice('\u3000 \t\n')
        if rng.random() < 0.5:
            text += space
        else:
            text = space + text
        nosiy = True
    return text, nosiy


def add_spaces_inside(text, rng=random):
    """在文本内部随机插入空格、\n 和 \t"""
    nosiy = False
    max_space = max(3, len(text) // 20 + 1)
    for _ in range(rng.randint(1, max_space)):
        space = rng.choice('\u3000 \t\n')
        index = rng.choice(range(len(text) - 1))
        if text[index].isdigit() and text[index + 1].isdigit():
            continue
        text = text[:(index + 1)] + space + text[(index + 1):]
        nosiy = True
    return text, nosiy


def swap_adjacent_characters(text, rng=random):
    """随机交换两个连续字符"""
    nosiy = False
    chars = list(text)
    for _ in range(rng.randint(1, 2)):
        index = rng.choice(range(len(text) - 1))
        if any([
            check(chars[index]) and check(chars[index + 1])
            for check in (is_cjk_character, is_english_character)
        ]):
            chars[index], chars[index+1] = chars[index+1], chars[index]
            nosiy = True
    return ''.join(chars), nosiy


def repeat_characters(text, rng=random):
    """随机重复字符"""
    nosiy = False
    for _ in range(rng.randint(1, 2)):
        index = rng.choice(range(len(text) - 1))
        if any([
            check(text[index]) and check(text[index + 1])
            for check in (is_cjk_character, is_english_character)
        ]):
            text = text[:(index+1)] + text[index] + text[(index+1):]
            nosiy = True
    return text, nosiy


def change_punctuation_width(text, rng=random):
    """随机改变标点符号的全半角"""
    nosiy = False
    punc_indice = [
        i for i, char in enumerate(text)
        if char in punctuation_mapping
    ]
    if punc_indice:
        rng.shuffle(punc_indice)
        chars = list(text)
        for index in punc_indice[:rng.randint(1, 3)]:
            chars[index] = punctuation_mapping[chars[index]]
            nosiy = True
        text = ''.join(chars)
    return text, nosiy


def is_cjk_character(char):
    """是否是中日韩字符"""
    return re.match(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', char) is not None


def is_english_character(char):
    """是否为英文字符"""
    return re.match(r'[a-zA-Z]', char) is not None


punctuation_mapping = {
    # 英文标点符号映射到对应的全角形式
    "!": "！",
    "\"": "“",
    "#": "＃",
    "$": "＄",
    "%": "％",
    "&": "＆",
    "'": "‘",
    "(": "（",
    ")": "）",
    "*": "＊",
    "+": "＋",
    ",": "，",
    "-": "－",
    ".": "．",
    "/": "／",
    ":": "：",
    ";": "；",
    "<": "＜",
    "=": "＝",
    ">": "＞",
    "?": "？",
    "@": "＠",
    "[": "【",
    "\\": "＼",
    "]": "】",
    "^": "＾",
    "_": "＿",
    "`": "｀",
    "{": "｛",
    "|": "｜",
    "}": "｝",
    "~": "～",

    # 中文标点符号映射到对应的半角形式
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "“": "\"",
    "”": "\"",
    "‘": "'",
    "’": "'",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "、": "\\",
    "；": ";",
    "：": ":",
    "…": "...",
    "·": "·",
    "—": "-",
    "＄": "$",
    "﹃": "{",
    "﹄": "}",
    "【": "[",
    "】": "]",
    "『": "{",
    "』": "}",
    "「": "{",
    "」": "}",
    "﹃": "{",
    "﹄": "}"
}


if __name__ == '__main__':
    print(add_noise('这是一个示例文本，包含中英文标点和数学公式。Hello, world! 注意保护公式x^2+2x=123。'))
