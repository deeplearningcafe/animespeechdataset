from fastapi import FastAPI, HTTPException, Query, Request, status, UploadFile
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path, audio_from_numpy
from reazonspeech.nemo.asr.interface import TranscribeResult

from fastapi.responses import FileResponse, Response
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import torch
import os

def transcribe_audio(audio_path:str=None, device:str="cuda") -> TranscribeResult:
    
    model = load_model(device=device)

    # ローカルの音声ファイルを読み込む
    audio = audio_from_path(audio_path)
    print(audio)

    autocast = torch.cuda.amp.autocast
    # 音声認識を適用する
    with autocast(dtype=torch.float16):
        with torch.no_grad():
            ret = transcribe(model, audio)
    print(ret)
    # with only the segments, json works good
    # returning everything works as well
    return ret.segments




if __name__ == "__main__":
    
    tmp_file_dir = "/tmp/example-files"
    os.makedirs(tmp_file_dir, exist_ok=True)

    app = FastAPI()

    @app.post(
        path="/api/media-file",
        # probably this should be updated
        response_class=JSONResponse,
    )
    async def extract_subtitles(file: UploadFile):
        """
        Receive File, store to disk & return it
        """
        print(file.filename)
        # Write file to disk. This simulates some business logic that results in a file sotred on disk
        audio_path = os.path.join(tmp_file_dir, file.filename)
        with open(audio_path, 'wb') as disk_file:
            file_bytes = await file.read()

            disk_file.write(file_bytes)

            result = transcribe_audio(audio_path, "cuda")
            # result = result.to("cpu")
            # print(f"Received file named {file.filename} containing {len(file_bytes)} bytes. ")

            # return FileResponse(disk_file.name, media_type=file.content_type)
            # return Response(content=result)
            # return result
            json_compatible_item_data = jsonable_encoder(result)
            torch.cuda.empty_cache()
            return JSONResponse(content=json_compatible_item_data)



    uvicorn.run(
        app, port=8001, host="0.0.0.0", log_level="warning"
    )






