from fastapi import FastAPI, HTTPException, Query, Request, status, UploadFile
from fastapi.responses import FileResponse, Response
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List, Union
from pydantic import BaseModel, Field

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

import os
import numpy as np
from espnet2.bin.spk_inference import Speech2Embedding


class Embedding(BaseModel):
    """Class for embedding"""
    embedding: List[float] = Field(max_length=192)

class EmbeddingResponse(BaseModel):
    """Class for embedding response"""
    data: List[Embedding]


def extract_embeddings(audio_paths:list[str], model):
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


if __name__ == "__main__":
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
        # print(audio_paths)
        print(audios_processing)
        # audio, fs = librosa.load(audios_processing[0])
        # print(audio)
        # audio = preprocess_audio()
        
        model = load_model("espnet/voxcelebs12_ecapa_wavlm_joint", "cuda")
        result = extract_embeddings(audios_processing, model)
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



    uvicorn.run(
        app, port=8001, host="0.0.0.0", log_level="debug"
    )
