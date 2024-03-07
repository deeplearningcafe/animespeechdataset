import demoji
import neologdn
import re

def convert_time(timestamp:str=None) -> float:
    """Convert time of this format 00:01:42,930, to seconds and miliseconds.

    Args:
        timestamp (str, optional): time in str. Defaults to None.

    Returns:
        float: seconds from the time str
    """
    parts = timestamp.split(',')
    hours_minutes_seconds = parts[0].split(':')
    milliseconds = parts[1]

    hours = float(hours_minutes_seconds[0])
    minutes = float(hours_minutes_seconds[1])
    seconds = float(hours_minutes_seconds[2])

    total_seconds = hours * 3600 + minutes * 60 + seconds + float(milliseconds) / 1000.0
    return total_seconds

def time_to_seconds(time):
    # 時間の各部分を取得
    hours, minutes, seconds, milliseconds = map(int, time.split('.'))

    # 時間を秒に変換
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds

def extract_main_text(line:str=None) -> str:
    """Normalize text

    Args:
        line (str, optional): text to normalize. Defaults to None.

    Returns:
        str: cleaned text
    """
    text = line.strip()
    # 文字ごとに分割します
    text = list(filter(lambda line: line != '', text))
    
    text = ''.join(text)

    text = text.translate(str.maketrans(
        {'\n': '', '\t': '', '\r': '', '\u3000': '', '《': '', '》': ''}
    ))

    # URLの消去
    text = re.sub(r'http?://[\w/:%#$&\?~\.=\+\-]+', '', text)
    text = re.sub(r'https?://[\w/:%#$&\?~\.=\+\-]+', '', text)

    text = demoji.replace(string=text, repl='')

    # 文字の正規化
    text = neologdn.normalize(text)

    #　数字をすべて0に
    text = re.sub(r'\d+', '0', text)

    # 大文字を小文字に
    text = text.lower()
    
    # 括弧とその中のテキストを除去する正規表現パターン
    pattern = r"\([^()]*\)"
    
    # 正規表現を使って括弧とその中のテキストを除去
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text