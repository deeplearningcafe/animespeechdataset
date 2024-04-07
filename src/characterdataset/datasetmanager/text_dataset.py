import csv
import os
from tqdm import tqdm
import pandas as pd

from .utils import time_to_seconds, extract_main_text, convert_time
from ..common import log


def parse_lines(lines:list=None) -> list[list[float, float, str]]:
    """From a list of lines of a str file, get a list with elements containing (start_time, end_time, dialogue)

    str format is like this:
        2
        00:01:42,930 --> 00:01:48,600
        《レイ：魔法とは　この世界における
        最先端技術である。
    Returns:
        start_time,end_time,text
        3.4,7.57,﻿0レイ:王立学院を舞台に0人の王子様との恋を楽しむ

    
    Args:
        lines (list, optional): _description_. Defaults to None.

    Returns:
        list[[float, float, str]]: a list containing the formated lines of the subtitle file.
    """
    # 最初色んな変数の定義をします
    dialogues = []
    start_time = None
    end_time = None
    current_dialogue = []
    
    for line in tqdm(lines, 'formatting the str file'):
        # ブランクを白状します
        line = line.strip()
        
        if line.isdigit():
            # 新しいラインなら時間数があるなら全部のリストに格納する
            if start_time is not None and end_time is not None:
                dialogues.append((start_time, end_time, "".join(current_dialogue)))
                current_dialogue = []
            
        elif '-->' in line:
            # 時間数を格納する
            start, end = line.split('-->')
            start = start.strip()
            end = end.strip()
            start_time = convert_time(start)
            end_time = convert_time(end)
        
        elif line:
            # 文章を格納する
            current_dialogue.append(extract_main_text(line))
    
    # 最後のラインの場合です        
    if start_time is not None and end_time is not None:
        dialogues.append((start_time, end_time, "".join(current_dialogue)))
        
    return dialogues

def str_2_csv(input_path:str=None, output_path:str=None, cropping:bool=False, num_characters:int=4) -> str:
    """Given the path of a str file, outputs the csv cleaned version. In case of use for labeling, it adds the character column

    Args:
        path (str, optional): path of the str file. Defaults to None.
        output_path (str, optional): path to output the csv file. Defaults to None.
        cropping (bool, optional): if the file is going to be used for cropping,
        then we want to include the column of characters. Defaults to False.
        num_characters (int, optional): min length of the text. Defaults to 4.


    """
    with open(input_path, 'r', encoding='utf-8') as file:
        # 行のリストを格納する
        lines = file.readlines()
        
        dialogues = parse_lines(lines)
    file.close()
    
    filename = os.path.basename(input_path)
    filename, format = os.path.splitext(filename)
    
    csv_filename = f"{output_path}/{filename}.csv"
    # CSVファイルにデータを書き込む
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if cropping:
            writer.writerow(['character', 'start_time', 'end_time', "text"])  # ヘッダーを書き込む
    
            for i in range(len(dialogues)):
                if len(dialogues[i][2]) > num_characters:
                    blank = ['']
                    blank.extend(dialogues[i])
                    writer.writerow(blank)  # 行を書き込む
        else:
            writer.writerow(['start_time', 'end_time', "text"])  # ヘッダーを書き込む
            for i in range(len(dialogues)):
                if len(dialogues[i][2]) > num_characters:
                        writer.writerow(dialogues[i])  # 行を書き込む
        
    log.info(f'CSVファイル "{csv_filename}" にデータを保存しました。')
    return csv_filename

def conversation_dataset(dialogues: pd.DataFrame=None, time_interval:int=3, first_characters:list=None, second_characters:list=None) -> list[list[str], list[str]]:
    """Given a dataframe with columns ["filename", "predicted_label", "start_time", "end_time", "text"],
    returns a conversational like list where all the user interactions are from the first character and 
    all the system interactions are from the second character.

    Args:
        dialogues (pd.DataFrame, optional): dataframe with the texts and the timings. Defaults to None.
        time_interval (int, optional): time between lines of the script. Defaults to 3.
        first_character (str, optional): in the dialog always the same character as user. Defaults to None.
        second_character (str, optional): in the dialog always the same character as assistant. Defaults to None.

    Returns:
        list[list[str], list[str]]: returns 2 lists one for the user interactions and other for the assistnant responses
    """
    user_lines = []
    assistant_lines = []
    
    for i in tqdm(range(len(dialogues) - 1)):
        current_dialogue = dialogues.iloc[i]
        next_dialogue = dialogues.iloc[i+1]
        
        # the predicted label column is the name of the character
        if current_dialogue[1] in first_characters and next_dialogue[1] in second_characters:
            # compare end of actual dialog with start of next dialog
            if next_dialogue[2] - current_dialogue[3] <= time_interval:
                user_lines.append(extract_main_text(current_dialogue[4]))
                assistant_lines.append(extract_main_text(next_dialogue[4]))
    
    return user_lines, assistant_lines

def clean_csv(csv_path:str=None, num_characters:int=4) -> pd.DataFrame:
    """Given a csv with columns [pkl_filename, predicted_label, discance] outputs a 
    cleaned dataframe with columns ["filename", "predicted_label", "start_time", "end_time", "text"]

    Args:
        csv_path (str, optional): input file path. Defaults to None.
        num_characters (int, optional): min length of the text. Defaults to 4.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    # dfを読み込む
    df = pd.read_csv(csv_path, header=0)
    # キャラのいないサンプルを排除します
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # 色んなリストを初期化する
    start_times = []
    end_times = []
    texts = []
    drop_idx = []
    
    for i in tqdm(range(len(df)), "generate text and times from title"):
        # キャラの名前がいない場合、NANとしてある。
        file_name = df.iloc[i, 0]
        
        file = os.path.basename(file_name)
        # format is .pkl so 4 letters
        id_str = file[:-4]
        index, start_time, end_time, text = id_str.split('_')
        
        # 短い文章を無視する
        if len(text) > num_characters:
            start_time = time_to_seconds(start_time)
            end_time = time_to_seconds(end_time)
            
            start_times.append(start_time)
            end_times.append(end_time)
            texts.append(text)
            
        else:
            drop_idx.append(i)
            
    # dfの処理をします
    df = df.drop(index=drop_idx)
    df = df.reset_index(drop=True)
    df["start_time"] = start_times
    df["end_time"] = end_times
    df["text"] = texts
    df = df[["filename", "predicted_label", "start_time", "end_time", "text"]]
    log.info(df.head())
    return df



def dialoges_from_csv(csv_path:str=None, output_path:str=None, time_interval:int=3, num_characters:int=4,
                      first_character:str=None, second_character:str=None) -> None:
    """Given a csv with columns [pkl_filename, predicted_label, discance] outputs a conversational 
    like file, with format user, assistant.

    Args:
        csv_path (str, optional): input file path. Defaults to None.
        output_path (str, optional): output directory path. Defaults to None.
        time_interval (int, optional): time between lines of the script. Defaults to 3.
        num_characters (int, optional): min length of the text. Defaults to 4.
        first_character (str, optional): in the dialog always the same character as user. Defaults to None.
        second_character (str, optional): in the dialog always the same character as assistant. Defaults to None.
    """
    
    df = clean_csv(csv_path, num_characters)
    
    # check if more than one character
    if first_character != None:
        first_characters = first_character.split(",")
        first_characters = [character.strip() for character in first_characters]

        # add (可能) to the characters
        first_characters_kanou = ['（可能）' + character for character in first_characters]
        first_characters.extend(first_characters_kanou)
    
    if second_character != None:
        second_characters = second_character.split(",")
        second_characters = [character.strip() for character in second_characters]
        
        # add (可能) to the characters
        second_characters_kanou = ['（可能）' + character for character in second_characters]
        second_characters.extend(second_characters_kanou)
    
    if (first_character == None or len(first_character) < 1) and (second_character == None or len(second_character) < 1):
        first_characters = list(set(df.iloc[:, 1]))
        second_characters = first_characters
    
    elif first_character == None or len(first_character) < 1:
        first_characters = list(set(df.iloc[:, 1]))
    elif second_character == None or len(second_character) < 1:
        second_characters = list(set(df.iloc[:, 1]))
    
    log.info(f"Characters used for the dialogs are {first_characters} and {second_characters}")
    
    user_lines, assistant_lines = conversation_dataset(df, time_interval, first_characters, second_characters)


    filename = os.path.basename(csv_path)
    filename, format = os.path.splitext(filename)
    
    csv_filename = f"{output_path}/{filename}_dialogs.csv"

    # 2つのリストを組み合わせた行のリストを作成
    rows = zip(user_lines, assistant_lines)
    # CSVファイルにデータを書き込む
    with open(csv_filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['User', 'Assistant'])  # ヘッダーを書き込む
        writer.writerows(rows)  # 行を書き込む

    log.info(f'CSVファイル "{csv_filename}" にデータを保存しました。')
    
    

# parse segments from asr
def segments_2_annotations(segments: list[dict], file_path:str=None, num_characters:int=4, iscropping:bool=False) -> None:
    """From the segments of transcribing audio, it creates a csv file with the start time, end time and
    text of each line in the audio. In the case of using for labeling, iscropping option adds the character column

    Args:
        segments (list[dict]): _description_
        file_path (str, optional): _description_. Defaults to None.
        num_characters (int, optional): _description_. Defaults to 4.
        iscropping (bool, optional): _description_. Defaults to False.
    """
    start_times = []
    end_times = []
    texts = []
    
    for segment in tqdm(segments, "convert segments to annotations"):
        start = segment['start_seconds']
        end = segment['end_seconds']
        text = segment['text']

        if len(text) > num_characters:
            
            start_times.append(round(start, 3))
            end_times.append(round(end, 3))
            texts.append(text)
    
    df = pd.DataFrame({ 'start_time': start_times, 'end_time': end_times, "text": texts})
    
    if iscropping:
        df['character'] = []
        
        # reorder to have ['character', 'start_time', 'end_time', "text"]
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
    
    df.to_csv(file_path, index=False)
    log.info(f"CSV created at {file_path} with {len(df)} elements!")
    log.info("Completed")
    
    
def create_cleaning(csv_path:str) -> pd.DataFrame:
    """Creates a dataframe from the predictions csv, the columns are file name(not path),
    the text, the predicted label and the distance.

    Args:
        csv_path (str): path of the predictions csv file

    Returns:
        pd.DataFrame: dataframe removing the whole path of the embeddings files and creating a
        column for the text.
    """
    # dfを読み込む
    df = pd.read_csv(csv_path, header=0)
    # キャラのいないサンプルを排除します
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # 色んなリストを初期化する
    names = []
    texts = []
    
    for i in tqdm(range(len(df)), "generate text from title"):
        # キャラの名前がいない場合、NANとしてある。
        file_name = df.iloc[i, 0]
        
        file = os.path.basename(file_name)
        # format is .pkl so 4 letters
        id_str = file[:-4]
        index, start_time, end_time, text = id_str.split('_')
        
        # 短い文章を無視する
        texts.append(text)
        names.append(file)
                            
    # dfの処理をします
    df["text"] = texts
    df["filename"] = names
    df = df[["filename", "text", "predicted_label", "distance"]]
    log.info(df.head())
    return df

def update_predictions(prediction_path:str, cleaning_path:str) -> str:
    """ From the cleaning csv file, updates the filenames of the audios and the embeddings. To match the updated
    names, a new prediction file is created, with the filenames updated.

    Args:
        prediction_path (str): path to the original prediction csv.
        cleaning_path (str): path to the file with the cleaned texts.

    Returns:
        str: file path of the resulting predictions csv.
    """
    # dfを読み込む
    df = pd.read_csv(cleaning_path, header=0)
    # キャラのいないサンプルを排除します
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    predictions = pd.read_csv(prediction_path, header=0)
    # キャラのいないサンプルを排除します
    predictions = predictions.dropna()
    predictions = predictions.reset_index(drop=True)
    folder_path = os.path.dirname(cleaning_path)
    
    # 色んなリストを初期化する
    embeddings_files = os.listdir(f"{folder_path}/embeddings")
    voice_files = os.listdir(f"{folder_path}/voice")
    prediction_files = predictions["filename"]

    # normalize all the paths in the pandas series to be able to compare with the original 
    # embeddings file
    prediction_files = prediction_files.map(lambda p: os.path.normpath(p))
    keep_idx = []
    
    for i in tqdm(range(len(df)), "generate text from title"):
        # キャラの名前がいない場合、NANとしてある。
        file_name_embeds = df.iloc[i, 0]
        text = df.iloc[i, 1]
        character_label = df.iloc[i, 2]
        
        file = os.path.basename(file_name_embeds)
        filename, format = os.path.splitext(file)
        file_name_audio = f"{filename}.wav"
        
        # format is .pkl so 4 letters
        id_str = file[:-4]
        index, start_time, end_time, text_old = id_str.split('_')
        
        # udpate name of file changing only the text part
        name = f'{index}_{start_time}_{end_time}_{text}'.replace(':', '.')
        
        # embeddings
        embeds_name = f'{name}.pkl'
        if file_name_embeds in embeddings_files:
            file_name_embeds_complete = f"{folder_path}/embeddings/{file_name_embeds}"
            embeds_name = f"{folder_path}/embeddings/{embeds_name}"
            # if same name it shouldn't be a problem so no need to check
            os.rename(file_name_embeds_complete, embeds_name)
            
            # looks for the index of embeddings file that has been renamed to rename also the
            # predictions filepath
            idx = prediction_files[prediction_files==os.path.normpath(file_name_embeds_complete)].index[0]
            predictions.iloc[idx, 0] = embeds_name
            # update the character label as well
            predictions.iloc[idx, 1] = character_label
            keep_idx.append(idx)
            
        # audios
        voice_name = f'{name}.wav'
        if file_name_audio in voice_files:
            file_name_audio_complete = f"{folder_path}/voice/{file_name_audio}"
            voice_name = f"{folder_path}/voice/{voice_name}"
            # if file_name_audio_complete != voice_name:
            os.rename(file_name_audio_complete, voice_name)
            # log.info(f"Changed file {file_name_audio} to {voice_name}")


    predictions_file = os.path.basename(prediction_path)
    predictions_filename, format = os.path.splitext(predictions_file)
    predictions_name = f"{folder_path}/{predictions_filename}_cleaned.csv"
    predictions = predictions[predictions.index.isin(keep_idx)]
    predictions.to_csv(predictions_name, index=False)
    log.info(f"CSV created at {predictions_name} with {len(predictions)} elements!")
    return predictions_name