import csv
import pandas as pd
import os
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import os
import shutil

from .utils import time_to_seconds, extract_main_text, convert_time
from common.log import log

from .text_dataset import str_2_csv as subtitle_2_csv
from .text_dataset import dialoges_from_csv as csv_2_dialoges
from .audio_dataset import character_audios as csv_2_audios

# log_filename = "common\logs\dataset_manager.log"
# os.makedirs(os.path.dirname(log_filename), exist_ok=True)

# logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.DEBUG, format="%(asctime)s %(levelname)-7s %(message)s")

# log = logging.getLogger(__name__)




def conversation_dataset_first_char(dialogues: pd.DataFrame=None, time_interval:int=3, first_character:str=None) -> list[list[str], list[str]]:
    """Given a dataframe with columns ["filename", "predicted_label", "start_time", "end_time", "text"],
    returns a conversational like list where all the user interactions are from the first character

    Args:
        dialogues (pd.DataFrame, optional): dataframe with the texts and the timings. Defaults to None.
        time_interval (int, optional): time between lines of the script. Defaults to 3.
        first_character (str, optional): in the dialog always the same character as user. Defaults to None.

    Returns:
        list[list[str], list[str]]: returns 2 lists one for the user interactions and other for the assistnant responses
    """
    user_lines = []
    assistant_lines = []
    for i in tqdm(range(len(dialogues) - 1)):
        current_dialogue = dialogues.iloc[i]
        next_dialogue = dialogues.iloc[i+1]
        
        # the predicted label column is the name of the character
        if current_dialogue[1] == first_character:
            # compare end of actual dialog with start of next dialog
            if next_dialogue[2] - current_dialogue[3] <= time_interval:
                user_lines.append(extract_main_text(current_dialogue[4]))
                assistant_lines.append(extract_main_text(next_dialogue[4]))
    
    return user_lines, assistant_lines

def conversation_dataset(dialogues: pd.DataFrame=None, time_interval:int=3, first_characters:list=None, second_characters:list=None) -> list[list[str], list[str]]:
    """Given a dataframe with columns ["filename", "predicted_label", "start_time", "end_time", "text"],
    returns a conversational like list where all the user interactions are from the first character

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
     

def parse_lines(lines:list=None) -> list[list[float, float, str]]:
    """From a list of lines of a str file, get a list with elements containing (start_time, end_time, dialogue)

    str format is like this:
        2
        00:01:42,930 --> 00:01:48,600
        《レイ：魔法とは　この世界における
        最先端技術である。
    
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

def str_2_csv(input_path:str=None, output_path:str=None) -> None:
    """Given the path of a str file, outputs the csv cleaned version

    Args:
        path (str, optional): path of the str file. Defaults to None.
        output_path (str, optional): path to output the csv file. Defaults to None.
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
        writer.writerow(['start_time', 'end_time', "text"])  # ヘッダーを書き込む
        writer.writerows(dialogues)  # 行を書き込む
        
    log.info(f'CSVファイル "{csv_filename}" にデータを保存しました。')


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
    if second_character != None:
        second_characters = second_character.split(",")
        second_characters = [character.strip() for character in second_characters]
    
    if first_character == None and second_character == None:
        first_characters = list(set(df.iloc[:, 1]))
        second_characters = first_characters
    
    elif first_character == None:
        first_characters = list(set(df.iloc[:, 1]))
    elif second_character == None:
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
    

def audio_dataset_from_csv(csv_path:str=None, character:str=None, num_characters:int=4) -> pd.DataFrame:
    """Given a csv with columns [pkl_filename, predicted_label, discance] outputs a 
    cleaned dataframe with columns ["filename", "predicted_label", "start_time", "end_time", "text"],
    only includings lines from the character

    Args:
        csv_path (str, optional): input file path. Defaults to None.
        character (str, optional): _description_. Defaults to None.
        num_characters (int, optional): min length of the text. Defaults to 4.

    Returns:
        pd.DataFrame: _description_
    """
    # dfを読み込む
    df = pd.read_csv(csv_path, header=0)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # 色んなリストを初期化する
    start_times = []
    end_times = []
    texts = []
    drop_idx = []
    
    # 日本語のキーボードの（）と普通の()は違う
    character_list = [character, '（可能）' +character]
    log.info(character_list)
    
    for i in tqdm(range(len(df)), "generate text and times from title"):
        # キャラの名前がいない場合、NANとしてある。
        name = df.iloc[i, 1].strip()
        if name in character_list:

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
            
        else:
            print(name)
            drop_idx.append(i)
            
            
    # dfの処理をします
    df = df.drop(index=drop_idx)
    df = df.reset_index(drop=True)
    df["start_time"] = start_times
    df["end_time"] = end_times
    df["text"] = texts
    df = df[["filename", "predicted_label", "start_time", "end_time", "text"]]
    log.info("Dataframe created")
    return df
    
def copy_file(input_path:str=None, output_path:str=None, index:str=None) -> str:
    """Copies a given file to a given location and returns the new file path

    Args:
        input_path (str, optional): path of the file. Defaults to None.
        output_path (str, optional): directory to copy the file. Defaults to None.

    Returns:
        str: new file path
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    new_filename = index + '.wav'
    # ファイルをコピーして、名前を変更する
    shutil.copy2(input_path, os.path.join(output_path, new_filename))
    
    copied_file = os.path.join(output_path, new_filename)
    return copied_file
    
    

def character_audios(csv_path:str=None, character:str=None, num_characters:int=4, output_path:str=None, audios_path:str=None) -> None:
    """From a dataframe with the audio files of the character, copy the audio files to
    a new folder and return a csv file with the new file names and their text.

    Args:
        csv_path (str, optional): input file path. Defaults to None.
        character (str, optional): name of the character to take the audios. Defaults to None.
        num_characters (int, optional): min length of the text. Defaults to 4.
        output_path (str, optional): output directory. Defaults to None.
        audios_path (str, optional): directory were the original audio files are stored. Defaults to None.
    """
    df = audio_dataset_from_csv(csv_path, character, num_characters)
    
    audio_list = os.listdir(audios_path)
    filenames = df["filename"]
    new_names = []
    texts = []
    audio_output_path = f"{output_path}/{character}"
    
    for file in filenames:
        file_base = os.path.basename(file)
        
        # .pkl
        id_str = file_base[:-4]
        index, start_time, end_time, text = id_str.split('_')
        
        for i in range(len(audio_list)):
            file_base_audio = os.path.basename(audio_list[i])
            # .wav
            id_str = file_base_audio
            
            index_audio, start_time, end_time, _ = id_str.split('_')
            
            if index == index_audio:
                new_file = copy_file(os.path.join(audios_path, audio_list[i]), audio_output_path, index_audio)
                new_names.append(new_file)
                texts.append(text)
    
    log.info(f"All file copied at {audio_output_path}")
    
    df = pd.DataFrame({"filename": new_names, "text": texts})
    df_out = os.path.join(output_path, "text.list")
    df.to_csv(df_out, index=False)
    log.info(f"CSV created at {df_out}!")
    log.info("Completed")
            

def dataset_manager(args):
    
    # checking if output_folder is a folder
    if not os.path.isdir(args.output_path):
        log.warning('output_path does not exist')
        # create output_folder
        os.mkdir(args.output_path)
        log.warning(f'created folder at {args.output_path}')
    
    if args.dataset_type == "subtitles":
        # checking if subtitles_file is a file
        if not os.path.isfile(args.subtitles_file):
            log.warning('subtitles_file does not exist')
            return
        log.info("Starting to create subtitles file")
        str_2_csv(intput_path=args.subtitles_file, output_path=args.output_path)
    
    elif args.dataset_type == "dialogues":
        # checking if input_srt is a file
        if not os.path.isfile(args.annotation_file):
            log.warning('annotation_file does not exist')
            return
        
        if args.num_characters < 1 or args.num_characters == None:
            log.warning('num_characters must be >= 1')
            args.num_characters = 4
        if args.time_interval < 1 or args.time_interval == None:
            log.warning('time_interval must be >= 1')
            args.time_interval = 5

            
        log.info("Starting to create dialogues file")

        dialoges_from_csv(csv_path=args.annotation_file, output_path=args.output_path,
                          time_interval=args.time_interval, num_characters=args.num_characters,
                          first_character=args.first_character, second_character=args.second_character)
        
    elif args.dataset_type == "audios":
        if not os.path.isfile(args.annotation_file):
            log.warning('annotation_file does not exist')
            return
        if not os.path.isdir(args.audios_path):
            log.warning('audios_path does not exist')
            return
        
        if args.num_characters < 1 or args.num_characters == None:
            log.warning('num_characters must be >= 1')
            args.num_characters = 4
        
        if args.character == None:
            log.warning('character does not exist')
            return

        log.info("Starting to audio files")

        character_audios(csv_path=args.annotation_file, character=args.character,
                         num_characters=args.num_characters, output_path=args.output_path,
                         audios_path=args.audios_path)
            
        
    else:
        log.warning("That function does not exist.")
        return
    
    log.info(f"The function {args.dataset_type} has been completed!")



class DatasetManager:
    """ This class stores the variables to call the functions, and performs the check of the inputs,
    it also included update functions to change its values.
    We want to store the variables as when creating the whole dataset,
    we will use the same variables, output_path, annotation_file, etc; a lot
    of times so it is better to store them.
    """
    
    dataset_types = ["subtitles", "dialogues", "audios"]
    
    def __init__(self,
                 dataset_type:str=None,
                 input_path:str=None,
                 subtitles_file:str=None,
                 annotation_file:str=None,
                 output_path:str=None,
                 num_characters:int=None,
                 time_interval:int=None,
                 first_character:str=None,
                 second_character:str=None,
                 character:str=None,
                 crop:bool=False) -> None:
        
        self.dataset_type = dataset_type
        self.input_path = input_path
        self.subtitles_file = subtitles_file
        self.annoation_file = annotation_file
        self.output_path = output_path
        self.num_characters = num_characters
        self.time_interval = time_interval
        self.first_character = first_character
        self.second_character = second_character
        self.character = character
        self.crop = crop
        
    def inputs_check(self) -> str:
        log.info("Checking inputs")
        # checking if output_folder is a folder
        if not os.path.isdir(self.output_path):
            log.warning('output_path does not exist')
            # create output_folder
            os.mkdir(self.output_path)
            log.warning(f'created folder at {self.output_path}')
        
        if self.dataset_type == "subtitles":
            # checking if subtitles_file is a file
            if not os.path.isfile(self.subtitles_file):
                log.warning('subtitles_file does not exist')
                return
            log.info("Starting to create subtitles file")
            # str_2_csv(intput_path=self.subtitles_file, output_path=self.output_path)
        
        elif self.dataset_type == "dialogues":
            # checking if input_srt is a file
            if not os.path.isfile(self.annotation_file):
                log.warning('annotation_file does not exist')
                return
            
            if self.num_characters < 1 or self.num_characters == None:
                log.warning('num_characters must be >= 1')
                self.num_characters = 4
            if self.time_interval < 1 or self.time_interval == None:
                log.warning('time_interval must be >= 1')
                self.time_interval = 5

                
            log.info("Starting to create dialogues file")

            # dialoges_from_csv(csv_path=self.annotation_file, output_path=self.output_path,
            #                 time_interval=self.time_interval, num_characters=self.num_characters,
            #                 first_character=self.first_character, second_character=self.second_character)
            
        elif self.dataset_type == "audios":
            if not os.path.isfile(self.annotation_file):
                log.warning('annotation_file does not exist')
                return
            if not os.path.isdir(self.audios_path):
                log.warning(f'audios_path {self.audios_path} does not exist')
                return
            
            if self.num_characters < 1 or self.num_characters == None:
                log.warning('num_characters must be >= 1')
                self.num_characters = 4
            
            if self.character == None:
                log.warning('character does not exist')
                return

            log.info("Starting to audio files")

            # character_audios(csv_path=self.annotation_file, character=self.character,
            #                 num_characters=self.num_characters, output_path=self.output_path,
            #                 audios_path=self.audios_path)
                
            
        else:
            log.warning("That function does not exist.")
            return
        
        # log.info(f"The function {self.dataset_type} has been completed!")
        return "Success"
    
    def create_csv(self, dataset_type:str="subtitles", subtitles_file:str=None, output_path:str=None,crop:bool=None):
        # Check inputs
        self.update_dataset_type(dataset_type)
        self.update_subtitles_file(subtitles_file)
        self.update_output_path(output_path)
        self.update_crop(crop)
        checks = self.inputs_check()
        
        
        # TODO: add min length to the subtitles as we have to label after
        if checks == "Success":
            filename = subtitle_2_csv(input_path=self.subtitles_file, output_path=self.output_path,
                           cropping=self.crop)
            return "Completado", filename

        return "Error", None
    
    
    def create_dialogues(self, dataset_type:str="dialogues",
                         annotation_file:str=None, 
                         output_path:str=None,
                         time_interval:str=None, 
                         num_characters:str=None,
                         first_character:str=None, 
                         second_character:str=None):
        # Check inputs
        self.update_dataset_type(dataset_type)
        self.update_output_path(output_path)
        self.update_time_interval(time_interval)
        self.update_annotation_file(annotation_file)
        self.update_num_characters(num_characters)
        self.update_first_character(first_character)
        self.update_second_character(second_character)

        checks = self.inputs_check()
        
        if checks == "Success":
            csv_2_dialoges(csv_path=self.annotation_file, output_path=self.output_path,
                            time_interval=self.time_interval, num_characters=self.num_characters,
                            first_character=self.first_character, second_character=self.second_character)

    def create_audio_files(self, dataset_type:str="audios",
                         annotation_file:str=None, 
                         output_path:str=None,
                         audios_path:str=None, 
                         num_characters:str=None,
                         character:str=None, 
                         ):

        # Check inputs
        self.update_dataset_type(dataset_type)
        self.update_annotation_file(annotation_file)
        self.update_output_path(output_path)
        self.update_audios_path(audios_path)
        self.update_num_characters(num_characters)
        self.update_character(character)



        checks = self.inputs_check()
        
        if checks == "Success":
            csv_2_audios(csv_path=self.annotation_file, character=self.character,
                            num_characters=self.num_characters, output_path=self.output_path,
                            audios_path=self.audios_path)


    def update_dataset_type(self, dataset_type:str=None):
        if dataset_type != None:
            self.dataset_type = dataset_type
        
    def update_subtitles_file(self, subtitles_file:str=None):
        if subtitles_file != None:
            self.subtitles_file = os.path.join(self.input_path, subtitles_file)
        
    def update_output_path(self, output_path:str=None):
        if output_path != None:
            self.output_path = output_path

    def update_annotation_file(self, annotation_file:str=None):
        if annotation_file != None:
            self.annotation_file = os.path.join(self.output_path, annotation_file)
    
    def update_time_interval(self, time_interval:int=None):
        if time_interval != None:
            self.time_interval = time_interval

    def update_num_characters(self, num_characters:int=None):
        if num_characters != None:
            self.num_characters = num_characters

    def update_first_character(self, first_character:str=None):
        if first_character != None:
            self.first_character = first_character
            
    def update_second_character(self, second_character:str=None):
        if second_character != None:
            self.second_character = second_character

    def update_audios_path(self, audios_path:str=None):
        if audios_path != None:
            self.audios_path = os.path.join(self.output_path, audios_path)

    def update_character(self, character:str=None):
        if character != None:
            self.character = character
        
    def update_crop(self, crop:bool=None):
        if crop != None:
            self.crop = crop
            
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='キャラ認識、会話データと音声合成データを作るために'
    )
    parser.add_argument('--dataset_type', default=None, type=str, 
                        required=True, choices=["subtitles", "dialogues", "audios"], help='埋め込みを作成するためのモデル')


    parser.add_argument('--subtitles_file', default=None, type=str,
                        required=False, help='キャラ認識したい動画のstr文字ファイル')
    
    parser.add_argument('--annotation_file', default=None, type=str,
                        required=False, help='せりふのタイミングとキャラ、キャラ認識の出力')
    parser.add_argument('--output_path', default=".datasets", type=str,
                        required=True, help='出力ディレクトリ')
    

    
    # 会話データの引数
    parser.add_argument('--num_characters', default=4, type=int,
                        required=False, help='出力ディレクトリ')
    parser.add_argument('--time_interval', default=5, type=int,
                        required=False, help='出力ディレクトリ')
    parser.add_argument('--first_character', default=None, type=str,
                        required=False, help='出力ディレクトリ')
    parser.add_argument('--second_character', default=None, type=str,
                        required=False, help='出力ディレクトリ')
    
    
    # 録音データの引数
    parser.add_argument('--audios_path', default=None, type=str,
                        required=False, help='出力ディレクトリ')
    parser.add_argument('--character', default=None, type=str,
                        required=False, help='出力ディレクトリ')
    
    args = parser.parse_args()
    parser.print_help()
    dataset_manager(args)
    
    