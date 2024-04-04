import pandas as pd
import os
from tqdm import tqdm
import torchaudio
import pickle
import argparse
import torch
import torchaudio.transforms as T
import torch.nn.functional as F
import requests
import json
import gc

from ..common import log

from .utils import (ffmpeg_extract_audio,
                    make_filename_safe,
                    get_subdir,
                    get_filename,
                    srt_format_timestamp,)
    

def load_model(model_name:str=None, device:str=None):
    if model_name == "speechbrain":
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                                    run_opts={"device": device},)
        return classifier
    
    elif model_name == "wavlm":
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
            
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv',
                                                                     cache_dir="pretrained_models")
        model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv',
                                                cache_dir="pretrained_models").to(device)

        model_dict = {"feature_extractor": feature_extractor, 
                      "model": model}
        return model_dict
    elif model_name == "espnet":
        return None
    else:
        raise ValueError(f"Model {model_name} is not included")
        



class data_processor:
    """This class is used to store functions to extract audios from video,
    to extract embeddings given characters, and to extract embeddings without annotations
    
    """
    def __init__(self, embeddings_extractor, model_type:str=None, device:str="cuda"):
        self.classifier = embeddings_extractor
        self.voice_dir = 'voice'
        self.model_type = model_type
        self.device = device

        
    def extract_audios_by_csv(self, annotate_csv:str=None, video_path:str=None, save_folder:str=None) -> None:
        """From a csv with format [character,start_time,end_time,text] and a video,
        clip audios given the subtitles. Create a folder for each character in save_folder directory

        Args:
            annotate_csv (str): path to the annotations csv file.
            video_path (str): path to the video file.
            save_folder (str): folder name to store audios.
        """
        srt_data = pd.read_csv(annotate_csv).iloc[:,:4]
        srt_data = srt_data.dropna()
        srt_list = srt_data.values.tolist()
        
        for index, (character, start_time, end_time, subtitle) in enumerate(tqdm(srt_list, 'video clip by csv file start')):
            audio_output = f'{save_folder}/voice/{character}'
            os.makedirs(audio_output, exist_ok=True)
            index = str(index).zfill(4)
            text = make_filename_safe(subtitle)
            start_time = float(start_time)
            end_time = float(end_time)
            
            ss = srt_format_timestamp(start_time)
            ee = srt_format_timestamp(end_time)
            
            
            name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

            audio_output = f'{audio_output}/{name}.wav'
            ffmpeg_extract_audio(video_path,audio_output,start_time,end_time)
            

    def extract_embeddings(self, save_folder:str=None) -> None:
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
                      # embedding = embedding.squeeze(0)
                      # embeddings = torch.tensor(embeddings, dtype=torch.float32)

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
                    resampled_waveform = self.preprocess_audio(pth)
                    if self.model_type == "wavlm":
                        inputs = self.classifier["feature_extractor"](resampled_waveform, padding=True, return_tensors="pt", sampling_rate=16000)
                        inputs = inputs.to(self.device)
                        embeddings = self.classifier["model"](**inputs).embeddings
                        embeddings = F.normalize(embeddings.squeeze(1), p=2, dim=1)
                    elif self.model_type == "speechbrain":  
                        embeddings = self.classifier.encode_batch(resampled_waveform)
                    else:
                        # as it is supposed to used batchs, we need to make the input a list
                        embeddings = self.request_embeddings([pth]) # [192]
                        embeddings = torch.tensor(embeddings, dtype=torch.float32)
                        
                    # 埋め込みを保存する
                    with open(f"{new_dir}/{file}.pkl", "wb") as f:
                        pickle.dump(embeddings.detach().cpu(), f)
                except Exception as e:
                    # here we want to continue saving other embeddings despite one failing
                    log.error(f"Error when saving the embeddings. {e}")
                    continue
        log.info("録音データから埋め込みを作成しました。")
        
        
    def preprocess_audio(self, audio_path:str=None) -> torch.Tensor:
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
    
    def extract_audios_by_subs(self, annotate_csv:str=None, video_path:str=None, 
                               temp_folder:str="tmp", iscropping:bool=False) -> None:
        """From a csv with format [start_time,end_time,text] and a video,
        clip audios given the subtitles. Create a folder for the video name in temp_folder directory

        Args:
            annotate_csv (str): path to the annotations csv file.
            video_path (str): path to the video file.
            temp_folder (str): folder name to store audios.
            iscropping (bool): in the case the file has the first column characters
        """
        df = pd.read_csv(annotate_csv, header=0)
        if not iscropping:
            df = df.dropna()
        file = os.path.basename(video_path)
        filename, format = os.path.splitext(file)
        os.makedirs(f'{temp_folder}/{filename}/{self.voice_dir}', exist_ok=True)
        log.info(f'{temp_folder}/{filename}/{self.voice_dir}')

        
        for index in tqdm(range(len(df)), f'cropping audio from video'):
            if iscropping:
                _, start_time, end_time, text = df.iloc[index, :]
            else:
                start_time, end_time, text = df.iloc[index, :]
            index = str(index).zfill(4)
            start_time = float(start_time)
            end_time = float(end_time)
            
            ss = srt_format_timestamp(start_time)
            ee = srt_format_timestamp(end_time)
            
            
            name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

            audio_output = f'{temp_folder}/{filename}/{self.voice_dir}/{name}.wav'
            ffmpeg_extract_audio(video_path,audio_output,start_time,end_time)
    
    @staticmethod        
    def extract_audios_for_labeling(annotate_csv:str=None, video_path:str=None, 
                               temp_folder:str="tmp", iscropping:bool=False) -> None:
        """From a csv with format [start_time,end_time,text] and a video,
        clip audios given the subtitles. Create a folder for the video name in temp_folder directory.
        Updates the csv file adding the column with the paths of the audios.

        As the folder and files should be deleted after labeling, the names can be simple.
        Args:
            annotate_csv (str): path to the annotations csv file.
            video_path (str): path to the video file.
            temp_folder (str): folder name to store audios.
            iscropping (bool): in the case the file has the first column characters
        """
        df = pd.read_csv(annotate_csv, header=0)
        if not iscropping:
            df = df.dropna()
        file = os.path.basename(video_path)
        filename, format = os.path.splitext(file)
        # should be already created in finder.py
        log.info(f'Saving audio files at {temp_folder}')

        file_names = []
        num_trials = 3
        drop_idx = []
        for i in tqdm(range(len(df)), "extracting audio clips from video"):
            if iscropping:
                _, start_time, end_time, text = df.iloc[i, :]
            else:
                start_time, end_time, text = df.iloc[i, :]
            index = str(i).zfill(4)
            start_time = float(start_time)
            end_time = float(end_time)
                        
            # make it simpler
            name = f'{index}_{text}'

            audio_output = f'{temp_folder}/{name}.wav'
            actual_trial = 0
            exists = os.path.exists(audio_output)
            while not exists and actual_trial < num_trials:
                ffmpeg_extract_audio(video_path,audio_output,start_time,end_time)
                actual_trial +=1
                exists = os.path.exists(audio_output)
            
            if exists:
                file_names.append(audio_output)
            else:
                log.info(f"Unable to crop {audio_output}")
                drop_idx.append(i)
            
        df = df.drop(index=drop_idx)
        # not log it because we want the program to stop here
        assert len(file_names) == len(df), "The lenght of file_names is not enough"
        df["filename"] = file_names
        df.to_csv(annotate_csv, index=False)
        log.info(f'CSVファイル "{annotate_csv}" にデータを保存しました。')


    def extract_embeddings_new(self, video_path:str=None, temp_folder:str="tmp") -> None:
        """From directory with character names as folder and their audio files,
        extract embeddings and save them in the same character folder under embeddings

        Args:
            save_folder (str, optional): _description_. Defaults to None.
        """
        
        # 録音データから埋め込みを作る
        file = os.path.basename(video_path)
        filename, format = os.path.splitext(file)

        temp_dir = f'{temp_folder}/{filename}'

        # これはリストを返り値
        voice_files = os.listdir(os.path.join(temp_dir, self.voice_dir))

        
        # 埋め込みを作成する
        for pth in tqdm(voice_files, f'extract {filename} audio features ,convert .wav to .pkl'):
            new_dir = os.path.join(temp_dir, 'embeddings')
            os.makedirs(new_dir, exist_ok=True)
            file = os.path.basename(pth)
            file, format = os.path.splitext(file)
            pth = os.path.join(temp_dir, self.voice_dir, pth)
            try:
                resampled_waveform = self.preprocess_audio(pth)
                    
                embeddings = self.classifier.encode_batch(resampled_waveform)
                # print(embeddings.shape) torch.Size([1, 1, 192])

                # 埋め込みを保存する
                with open(f"{new_dir}/{file}.pkl", "wb") as f:
                    pickle.dump(embeddings.detach().cpu(), f)
            except Exception as e:
                # here we want to continue saving other embeddings despite one failing
                log.error(f"Error when saving the new embeddings. {e}")
                continue


        log.info("録音データから埋め込みを作成しました。")
    
    @staticmethod
    def request_transcription(audio_path:str=None) -> dict:
        """Sends a request to the api to transcribe the audio

        Args:
            audio_path (str, optional): path of the audio file, should be normalized. Defaults to None.

        Returns:
            dict: dictionary in json format containing the segments and their timings.
        """
        url = 'http://127.0.0.1:8001/api/media-file'
        #    curl -X 'POST' \
        #   'http://127.0.0.1:8001/api/media-file' \
        #   -H 'accept: application/json' \
        #   -H 'Content-Type: multipart/form-data' \
        #   -F 'file=@output.wav;type=audio/wav'
        headers = {
            'accept': 'application/json',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }
        try:
            # we have a problem when reading files because of the \0
            files = {
                'file': (audio_path, open(audio_path, 'rb'), 'audio/wav'),
            }
        except Exception as e:
            print(f"When calling api {e}")

        response = requests.post('http://127.0.0.1:8001/api/media-file', headers=headers, files=files)

        if response.status_code == 200:
            # it is a bytes object so first decode, then we change to json
            response_json = response.content.decode("utf-8")
            data = json.loads(response_json)
            return data
        else:
            log.warning(f"Bad request {response.status_code}")

    def request_embeddings(self, audio_paths: list[str]) -> dict:
        """Sends a request to the api to transcribe the audio

        Args:
            audio_path (str, optional): path of the audio file, should be normalized. Defaults to None.

        Returns:
            dict: dictionary in json format containing the segments and their timings.
        """
        url = 'http://localhost:8001/api/media-file'
        # curl -X 'POST' \
        #   'http://localhost:8001/api/media-file' \
        #   -H 'accept: application/json' \
        #   -H 'Content-Type: multipart/form-data' \
        #   -F 'files=@0023_00.03.09.380_00.03.12.550_私の名前覚えていらっしゃいますか.wav;type=audio/wav' \
        #   -F 'files=@0024_00.03.12.550_00.03.17.060_んっ…。バカにしていますのレイ=テイラー!.wav;type=audio/wav'
        headers = {
            'accept': 'application/json',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }
        try:
            # we have a problem when reading files because of the \0
            # files = {
            #     'file': (audio_path, open(audio_path, 'rb'), 'audio/wav'),
            # }
            files = []
            for audio in audio_paths:
                file_tmp = ('files', (audio, open(audio, 'rb'), 'audio/wav'))
                files.append(file_tmp)
            # files = [
            # ('files', ('0023_00.03.09.380_00.03.12.550_私の名前覚えていらっしゃいますか.wav', open('0023_00.03.09.380_00.03.12.550_私の名前覚えていらっしゃいますか.wav', 'rb'), 'audio/wav')),
            # ('files', ('0024_00.03.12.550_00.03.17.060_んっ…。バカにしていますのレイ=テイラー!.wav', open('0024_00.03.12.550_00.03.17.060_んっ…。バカにしていますのレイ=テイラー!.wav', 'rb'), 'audio/wav')),
            # ]
        except Exception as e:
            print(f"When opening files for request {e}")

        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            # reponse is like: {"data":[{"embedding": []}, {"embedding": []}]
            response_json = response.content.decode("utf-8")
            response_json = json.loads(response_json)
            data = []
            for embedding in response_json["data"]:
                # print(embedding["embedding"])
                # the embedding is a list of floats with lenght 192
                data.append(embedding["embedding"])
            return data
        else:
            log.warning(f"Bad request {response.status_code}")

    def request_embeddings_creation(self, character_folder: str) -> dict:
        """Sends a request to the api to create embeddings of the characters

        Args:
            character_folder (str, optional): path of the directory with the audios of each character, should be normalized. Defaults to None.

        Returns:
            str: a string with the result
        """
        url = 'http://localhost:8001/api/embeddings'
        # curl -X 'POST' \
        # 'http://localhost:8001/api/embeddings?character_folder=data%2Fcharacter_embedds' \
        # -H 'accept: */*' \
        # -d ''
        headers = {
            'accept': '*/*',
            'content-type': 'application/x-www-form-urlencoded',
        }
        params = {
            'character_folder': str(character_folder),
        }
        response = requests.post(url, headers=headers, params=params)

        if response.status_code == 200:
            response_content = response.content.decode("utf-8")
            return response_content
        else:
            log.error(f"Bad request {response.status_code}")
            response.raise_for_status()


    def request_embeddings_creation_new(self, video_path:str, temp_folder:str="tmp") -> dict:
        """Sends a request to the api to create embeddings for prediction

        Args:
            video_path (str, optional): path of the video file, should be normalized. Defaults to None.
            temp_folder (str, optional): path of the directory to store the embeddings, should be normalized. Defaults to None.

        Returns:
            str: a string with the result
        """
        url = 'http://localhost:8001/api/embeddings-predict'
        # curl -X 'POST' \
        # 'http://localhost:8001/api/embeddings?character_folder=data%2Fcharacter_embedds' \
        # -H 'accept: */*' \
        # -d ''
        headers = {
            'accept': '*/*',
            'content-type': 'application/x-www-form-urlencoded',
        }
        params = {
            'video_path': str(video_path),
            'temp_folder': str(temp_folder),
        }
        response = requests.post(url, headers=headers, params=params)

        if response.status_code == 200:
            response_content = response.content.decode("utf-8")
            return response_content
        else:
            log.error(f"Bad request {response.status_code}")
            response.raise_for_status()



def crop_command(args):
    
    # if args.verbose:
    #     print('running crop')
    
    # check if annotate_map is a file
    if not os.path.isfile(args.annotate_map):
        log.info(f'annotate_map {args.annotate_map} does not exist')
        return

    # check if role_audios is a folder
    if not os.path.isdir(args.role_audios):
        log.info(f'role_audios {args.role_audios} does not exist')
        # create role_audios folder
        os.mkdir(args.role_audios)
        
    # data = pd.read_csv(args.annotate_map)
    # speechbrainのモデルを読み込む
    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
    #                                             run_opts={"device": "cuda"},)
    # video_pth = args.video_path
    
    # clip_audio_bycsv(args.annotate_map, video_pth, args.role_audios)
    
    # extract_pkl_feat(classifier, args.role_audios)
    
    classifier = load_model(args.model, args.device)
    processor = data_processor(classifier, args.model, args.device)
    # 録音データを格納する
    processor.extract_audios_by_csv(args.annotate_map, args.video_path, args.role_audios)
    
    # 埋め込みを生成する
    processor.extract_embeddings(args.role_audios)
    

def crop(annotation_file:str=None,
         output_path:str=None,
         video_path:str=None,
         model:str=None,
         device:str=None,
        ):
    """Creates embeddings and save them in each character folder. This data will be used for predicting as the labeled data.
    First extract audios and then extract embeddings from the audios.

    Args:
        annotation_file (str, optional): path of the annotation file. Defaults to None.
        output_path (str, optional): path to output the character embeddings. Defaults to None.
        video_path (str, optional): path of the video file. Defaults to None.
        model (str, optional): name of the model to use for extracting the embeddings. Defaults to None.
        device (str, optional): device to use. Defaults to None.
    """
   
        
    
    classifier = load_model(model, device)
    processor = data_processor(classifier, model, device)
    # 録音データを格納する
    processor.extract_audios_by_csv(annotation_file, video_path, output_path)
    
    # 埋め込みを生成する
    if model == "espnet":
        result = processor.request_embeddings_creation(output_path)
        log.info(result)
    else:
        processor.extract_embeddings(output_path)
        
        # delete the model to release memory
        try:
            del classifier
            gc.collect()
            torch.cuda.empty_cache()
        except:
            log.warning(f"Couldn't delete the {model} model")
    
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='動画とcsvファイルから埋め込みと録音データを取得する'
    )
    # parser.add_argument("verbose", type=bool, action="store")
    parser.add_argument('--annotate_map', default='dataset_recognize.csv', 
                        type=str, required=True, help='せりふのタイミングとキャラ')
    parser.add_argument('--role_audios', default='./role_audios', type=str,
                        required=True, help='出力を保存するためのフォルダ')
    parser.add_argument('--video_path', default=None, type=str, required=True,
                        help='録音データを取得するための動画')
    parser.add_argument('--model', default="speechbrain", type=str, required=True, choices=["speechbrain", "wavlm"],
                        help='埋め込みを作成するためのモデル')
    parser.add_argument('--device', default="cuda", type=str, required=False, choices=["cuda", "cpu"],
                    help='埋め込みを作成するためのモデル')

    args = parser.parse_args()
    parser.print_help()
    # print(args)
    crop_command(args)