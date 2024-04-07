from fastapi import FastAPI, HTTPException, Query, Request, status, UploadFile
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path, audio_from_numpy
from reazonspeech.nemo.asr.interface import TranscribeResult

from fastapi.responses import FileResponse, Response
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import gc
import shutil
from tqdm import tqdm
import pandas as pd

import torch
import os
from ..common import log
# from ..datasetmanager.text_dataset import segments_2_annotations

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
    print(ret.segments)
    # with only the segments, json works good
    # returning everything works as well
    return ret.segments

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
        start = segment.start_seconds
        end = segment.end_seconds
        text = segment.text

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



# if __name__ == "__main__":
def reazonspeech_api():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    

    tmp_file_dir = "/tmp/example-files"
    tmp_file_dir = os.path.join(current_dir, tmp_file_dir)

    try:
        shutil.rmtree(tmp_file_dir)
    except:
        print("Couldn't remove the tmp folder")
    os.makedirs(tmp_file_dir, exist_ok=True)

    app = FastAPI()

    @app.post(
        path="/api/media-file",
        # probably this should be updated
        response_class=JSONResponse,
    )
    async def extract_subtitles(file: UploadFile):
        """
        Receive File, store to disk, then it transcribes it returning the segmentations objects.
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


    @app.post(
    path="/api/transcribe-folder",
    response_class=Response,
    )
    async def transcribe_folder(audio_path: str, filename: str, num_characters: int, iscropping: bool):
        """
        Receive a audio path and extract the subtitles with their times, from there use the segments_2_annotations
        function to create the annotations file.
        Args:
            audio_path (str): path of the audio file to transcribe.
            file_path (str, optional): path for saving the annotations csv. Defaults to None.
            num_characters (int, optional): min number of characters to use the phrase. Defaults to 4.
            iscropping (bool, optional): if true add a column for labeling characters. Defaults to False.
        Returns:
            str: a string with the result
        """
        segments = transcribe_audio(audio_path, "cuda")
        # json_segments= jsonable_encoder(segments)
        
        # json_compatible_item_data = jsonable_encoder(result)
        segments_2_annotations(segments, filename, num_characters, iscropping)
        
        try:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except:
            log.warning("Couldn't delete the espnet model")

        # return JSONResponse(content=json_compatible_item_data)
        return Response(content="Video transcribed!")


    uvicorn.run(
        app, port=8080, host="0.0.0.0", log_level="debug"
    )






