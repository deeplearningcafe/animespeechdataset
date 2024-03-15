import os
import shutil
import pandas as pd
from tqdm import tqdm

from .utils import time_to_seconds
from common.log import log

def copy_file_old(input_path:str=None, output_path:str=None):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file = os.path.basename(input_path)
    # .pkl
    id_str = file[:-4]
    index, start_time, end_time, text = id_str.split('_')
    
    new_filename = index + '.wav'
    # ファイルをコピーして、名前を変更する
    shutil.copy2(input_path, os.path.join(output_path, new_filename))
    
    copied_file = os.path.join(output_path, new_filename)
    return copied_file
    
def character_audios_old(df_path:str=None, output_path:str=None, audios_path:str=None) -> list:
    df = pd.read_csv(df_path, header=0)
    audio_list = os.listdir(audios_path)
    
    filenames = df["filename"]
    new_names = []
    texts = []
    
    for file in filenames:
        file_base = os.path.basename(file)
        # .pkl
        id_str = file_base[:-4]
        index, start_time, end_time, text = id_str.split('_')
        
        for i in range(len(audio_list)):
            file_base_audio = os.path.basename(audio_list[i])
            # .wav
            id_str = file_base_audio[:-4]

            index_audio, start_time, end_time, _ = id_str.split('_')
            
            if index == index_audio:
                new_file = copy_file(os.path.join(audios_path, audio_list[i]), output_path)
                new_names.append(new_file)
                texts.append(text)
    
    df = pd.DataFrame({"filename": new_names, "text": texts})
    df_out = os.path.join(output_path, "text.list")
    df.to_csv(df_out, index=False)
    log.info("CSV created!")
    log.info("Completed")

def copy_file(input_path:str=None, output_path:str=None, index:str=None, text:str=None) -> str:
    """Copies a given file to a given location and returns the new file path

    Args:
        input_path (str, optional): path of the file. Defaults to None.
        output_path (str, optional): directory to copy the file. Defaults to None.
        index (str, optional): index for the name. Defaults to None.
        text (str, optional): text for the name. Defaults to None.
    Returns:
        str: new file path
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # update the names to include text, to add audios from other annotations
    new_filename = index + text + '.wav'
    # ファイルをコピーして、名前を変更する
    shutil.copy2(input_path, os.path.join(output_path, new_filename))
    
    copied_file = os.path.join(output_path, new_filename)
    return copied_file

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
    
    log.info(f"Looking for audios at {audios_path}")
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
                new_file = copy_file(os.path.join(audios_path, audio_list[i]), audio_output_path, index_audio, text)
                new_names.append(new_file)
                texts.append(text)
    
    log.info(f"All file copied at {audio_output_path}")
    
    df = pd.DataFrame({"filename": new_names, "text": texts})
    df_out = os.path.join(output_path, "text.list")
    df.to_csv(df_out, index=False)
    log.info(f"CSV created at {df_out} with {len(df)} elements!")
    # log.info("Completed")

    
    
# def main():
#     df_path = "datasets\wataoshi2_clair.csv"
#     output_path = "audio\clair"
#     audios_path = r"tmp\[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 02 [WebRip 1080p HEVC-10bit AAC ASSx2]\voice"
#     character_audios(df_path, output_path, audios_path)

# if __name__ == "__main__":
#     main()
    
