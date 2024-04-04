from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response
import uvicorn
from typing import List
from pydantic import BaseModel, Field
import gc

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import pickle
from tqdm import tqdm

from ..common import log
# import logging
# logging.getLogger("espnet2").setLevel(logging.WARNING)

import os
import numpy as np
from espnet2.bin.spk_inference import Speech2Embedding
from ..oshifinder.utils import (
    get_subdir,
    get_filename,
)

# from ..oshifinder.crop import crop



class Embedding(BaseModel):
    """Class for embedding"""
    embedding: List[float] = Field(max_length=192)

class EmbeddingResponse(BaseModel):
    """Class for embedding response"""
    data: List[Embedding]



def extract_embeddings_api(audio_paths:list[str], model):
    embeddings = []
    for audio_path in audio_paths:
        with torch.no_grad():
            torch.cuda.empty_cache()
            # torch.Tensorとnumpyを使える、これはtorch.Tensorです
            # print(audio_path)
            # if os.path.isfile(audio_path):
            #     print(True)
            audio_processed = preprocess_audio(audio_path)
             # ローカルの音声ファイルを読み込む
            embedding = model(audio_processed)
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.squeeze(0)
            embeddings.append(embedding.detach().cpu().numpy())
    
    print(len(embeddings), type(embeddings))

    # autocast = torch.cuda.amp.autocast
    # # 音声認識を適用する
    # with autocast(dtype=torch.float16):
    #     with torch.no_grad():
    #         ret = transcribe(model, audio)
    # print(ret)
    # with only the segments, json works good
    # returning everything works as well
    data = [Embedding(embedding=embedd) for embedd in embeddings]
    return data

def load_model(model_tag:str, device:str="cuda"):
    # from uploaded models
    speech2spk_embed = Speech2Embedding.from_pretrained(model_tag=model_tag, device=device)

    return speech2spk_embed

def preprocess_audio(audio_path:str=None) -> torch.Tensor:
    """Preprocessing of audios, convert to mono signal and resample

    Args:
        audio_path (str, optional): path of the audio file. Defaults to None.

    Returns:
        torch.Tensor: tensor containing the information of the audio file
    """
    # サンプリングレートは16khzであるべき
    signal, fs = torchaudio.load(audio_path)
    # 録音の前処理
    signal_mono = torch.mean(signal, dim=0)
    # change freq
    resample_rate = 16000
    resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
    resampled_waveform = resampler(signal_mono)
    
    return resampled_waveform


def extract_embeddings(model, save_folder:str) -> None:
    """From directory with character names as folder and their audio files,
    extract embeddings and save them in the same character folder under embeddings

    Args:
        save_folder (str, optional): _description_. Defaults to None.
    """
    
    # これはキャラの名前をとってる
    sub_dirs = get_subdir(f'{save_folder}/voice')
    
    for dir in sub_dirs:
        # これはリストを返り値
        voice_files = get_filename(dir)
        name = os.path.basename(os.path.normpath(dir))
        new_dir = os.path.join(save_folder, 'embeddings', name)
        os.makedirs(new_dir, exist_ok=True)
        
        for file, pth in tqdm(voice_files, f'extract {name} audio embeddings ,convert .wav to .pkl'):
            try:
                resampled_waveform = preprocess_audio(pth)

                embeddings = model(resampled_waveform) # [1, 192]
                embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                # 埋め込みを保存する
                with open(f"{new_dir}/{file}.pkl", "wb") as f:
                    pickle.dump(embeddings.detach().cpu(), f)
            except Exception as e:
                # here we want to continue saving other embeddings despite one failing
                log.error(f"Error when saving the embeddings. {e}")
                continue
        log.info(f"Extracted embeddings from {name}")
    # log.info("録音データから埋め込みを作成しました。")
    return "録音データから埋め込みを作成しました。"

def extract_embeddings_predict(model, video_path:str, temp_folder:str="tmp") -> None:
    """From directory with character names as folder and their audio files,
    extract embeddings and save them in the same character folder under embeddings

    Args:
        save_folder (str, optional): _description_. Defaults to None.
    """
    voice_dir = "voice"
    # 録音データから埋め込みを作る
    file = os.path.basename(video_path)
    filename, format = os.path.splitext(file)

    temp_dir = f'{temp_folder}/{filename}'

    # これはリストを返り値
    voice_files = os.listdir(os.path.join(temp_dir, voice_dir))
    new_dir = os.path.join(temp_dir, 'embeddings')
    os.makedirs(new_dir, exist_ok=True)
    
    # 埋め込みを作成する
    for pth in tqdm(voice_files, f'extract {filename} audio features ,convert .wav to .pkl'):
        file = os.path.basename(pth)
        file, format = os.path.splitext(file)
        pth = os.path.join(temp_dir, voice_dir, pth)
        try:
            resampled_waveform = preprocess_audio(pth)
                
            embeddings = model(resampled_waveform) # [1, 192]
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 埋め込みを保存する
            with open(f"{new_dir}/{file}.pkl", "wb") as f:
                pickle.dump(embeddings.detach().cpu(), f)
        except Exception as e:
            # here we want to continue saving other embeddings despite one failing
            log.error(f"Error when saving the new embeddings. {e}")
            continue


    return "録音データから埋め込みを作成しました。"

def main():
    # add logger
    current_dir = os.path.dirname(os.path.abspath(__file__))
    

    tmp_file_dir = "/tmp/example-files"
    tmp_file_dir = os.path.join(current_dir, tmp_file_dir)
    os.makedirs(tmp_file_dir, exist_ok=True)

    app = FastAPI()

    @app.post(
        path="/api/media-file",
        # probably this should be updated
        response_model=EmbeddingResponse,
    )
    async def get_embeddings(files: List[UploadFile]):
        """
        Receive File, store to disk & return it
        """
        # Write file to disk. This simulates some business logic that results in a file sotred on disk
        audios_processing = []
        for file in files:
            audio_path = os.path.join(tmp_file_dir, file.filename)
            with open(audio_path, 'wb') as disk_file:
                file_bytes = await file.read()

                disk_file.write(file_bytes)
                # print(disk_file)
                audios_processing.append(audio_path)
                
        audio_paths = os.listdir(tmp_file_dir)
        audio_paths = [os.path.join(tmp_file_dir, audio) for audio in audio_paths]
        print(audios_processing)

        model = load_model("espnet/voxcelebs12_ecapa_wavlm_joint", "cuda")
        result = extract_embeddings_api(audios_processing, model)

        # here we should delete the tmp files
        torch.cuda.empty_cache()
        # reponse is like: Data[{"embedding": []}, {"embedding": []}]
        return EmbeddingResponse(data=result)


    @app.post(
    path="/api/embeddings",
    response_class=Response,
    )
    async def create_embeddings(character_folder: str):
        """
        Receive a folder path were the voices of the characters are stored, and extract their embeddings
        Args:
            character_folder (str, optional): path of the directory with the audios of each character, should be normalized. Defaults to None.

        Returns:
            str: a string with the result
        """
        model = load_model("espnet/voxcelebs12_ecapa_wavlm_joint", "cuda")
        result = extract_embeddings(model, character_folder)
        
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            log.warning("Couldn't delete the espnet model")

        return Response(content=result)
    
    @app.post(
    path="/api/embeddings-predict",
    response_class=Response,
    )
    async def create_embeddings_predict(video_path:str, temp_folder:str="tmp"):
        """
        Sends a request to the api to create embeddings for prediction

        Args:
            video_path (str, optional): path of the video file, should be normalized. Defaults to None.
            temp_folder (str, optional): path of the directory to store the embeddings, should be normalized. Defaults to None.

        Returns:
            str: a string with the result
        """
        model = load_model("espnet/voxcelebs12_ecapa_wavlm_joint", "cuda")
        result = extract_embeddings_predict(model, video_path, temp_folder)
        
        # delete the model to release memory
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            log.warning("Couldn't delete the espnet model")

        return Response(content=result)
    
    uvicorn.run(
        app, port=8001, host="0.0.0.0", log_level="debug"
    )
