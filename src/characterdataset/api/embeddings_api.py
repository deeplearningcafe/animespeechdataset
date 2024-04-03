from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response
import uvicorn
from typing import List
from pydantic import BaseModel, Field

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
        
        # if self.model_type == "espnet":
        #     batch_size = 8
        #     num_batches = (len(voice_files) + batch_size - 1) // batch_size

        #     for i in tqdm(range(num_batches), f'extract {name} audio embeddings ,convert .wav to .pkl'):
        #         start_idx = i * batch_size
        #         end_idx = min((i + 1) * batch_size, len(voice_files))
        #         batch_paths = voice_files[start_idx:end_idx]
        #         batch_files = [file for file, pth in batch_files]
        #         batch_paths = [pth for file, pth in batch_files]

        #     try:
        #         embeddings = self.request_embeddings(batch_paths)
        #         for i, file in enumerate(batch_files):
        #             with open(f"{new_dir}/{file}.pkl", "wb") as f:
        #                 pickle.dump(embeddings[i], f)
            
        #     except Exception as e:
        #         # here we want to continue saving other embeddings despite one failing
        #         log.error(f"Error when saving the embeddings. {e}")
        #         continue

            
        # else:
        for file, pth in tqdm(voice_files, f'extract {name} audio embeddings ,convert .wav to .pkl'):
            try:
                resampled_waveform = preprocess_audio(pth)
                # if self.model_type == "wavlm":
                #     inputs = self.classifier["feature_extractor"](resampled_waveform, padding=True, return_tensors="pt", sampling_rate=16000)
                #     inputs = inputs.to(self.device)
                #     embeddings = self.classifier["model"](**inputs).embeddings
                #     embeddings = F.normalize(embeddings.squeeze(1), p=2, dim=1)
                # elif self.model_type == "speechbrain":  
                #     embeddings = self.classifier.encode_batch(resampled_waveform)
                # else:
                # as it is supposed to used batchs, we need to make the input a list
                # embeddings = request_embeddings([pth]) # [192]
                embeddings = model(resampled_waveform) # [1, 192]
                embeddings = F.normalize(embeddings, p=2, dim=1)
                # embedding = embedding.squeeze(0)
                # embeddings = torch.tensor(embeddings, dtype=torch.float32)
                    
                # 埋め込みを保存する
                with open(f"{new_dir}/{file}.pkl", "wb") as f:
                    pickle.dump(embeddings.detach().cpu(), f)
            except Exception as e:
                # here we want to continue saving other embeddings despite one failing
                log.error(f"Error when saving the embeddings. {e}")
                continue
        log.info("Extracted embeddings from {name}")
    log.info("録音データから埋め込みを作成しました。")
    return "Completed"

def main():
    # add logger
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the absolute path to the log file
    # log_file_path = os.path.join(current_dir, "embeddings_api.log")

    # logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    # logging.getLogger("espnet2").setLevel(logging.WARNING)

    # logger = logging.getLogger(__name__)
    
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)

    # # configure the handler and formatter as needed
    # api_handler = logging.FileHandler(f"{__name__}.log", mode='w')
    # papi_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    # # add formatter to the handler
    # api_handler.setFormatter(papi_formatter)
    # # add handler to the logger
    # logger.addHandler(api_handler)

    # logger.info(f"Testing the custom logger for module {__name__}...")


    
    # current_dir = os.path.dirname(os.path.abspath(__file__))

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
        # print(audio_paths)
        print(audios_processing)
        # audio, fs = librosa.load(audios_processing[0])
        # print(audio)
        # audio = preprocess_audio()
        
        model = load_model("espnet/voxcelebs12_ecapa_wavlm_joint", "cuda")
        result = extract_embeddings_api(audios_processing, model)
        # result = result.to("cpu")
        # print(f"Received file named {file.filename} containing {len(file_bytes)} bytes. ")

        # return FileResponse(disk_file.name, media_type=file.content_type)
        # return Response(content=result)
        # return result
            
        torch.cuda.empty_cache()
        # reponse is like: Data[{"embedding": []}, {"embedding": []}]
        return EmbeddingResponse(data=result)


    @app.post(
    path="/api/test-file",
    response_class=FileResponse,
    )
    async def post_media_file(file: UploadFile):
        """
        Receive File, store to disk & return it
        """
        # Write file to disk. This simulates some business logic that results in a file sotred on disk
        with open(os.path.join(tmp_file_dir, file.filename), 'wb') as disk_file:
            file_bytes = await file.read()

            disk_file.write(file_bytes)

            print(f"Received file named {file.filename} containing {len(file_bytes)} bytes. ")

            return FileResponse(disk_file.name, media_type=file.content_type)


    @app.post(
    path="/api/embeddings",
    response_class=Response,
    )
    async def create_embeddings(character_folder: str):
        """
        Receive File, store to disk & return it
        """
        model = load_model("espnet/voxcelebs12_ecapa_wavlm_joint", "cuda")
        result = extract_embeddings(model, character_folder)

        return Response(content=result)
    uvicorn.run(
        app, port=8001, host="0.0.0.0", log_level="debug"
    )
