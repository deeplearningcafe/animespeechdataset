import pandas as pd
import os
from tqdm import tqdm
import torchaudio
import pickle
import argparse
import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from common.log import log

from .utils import (ffmpeg_extract_audio,
                    make_filename_safe,
                    get_subdir,
                    get_filename,
                    srt_format_timestamp)

def clip_audio_bycsv(annotate_csv,video_pth,role_audios):
        annotate_csv = annotate_csv
        video_pth = video_pth
        role_audios = role_audios
        srt_data = pd.read_csv(annotate_csv).iloc[:,:4]
        srt_data = srt_data.dropna()
        srt_list = srt_data.values.tolist()
        for index, (person,start_time,end_time, subtitle) in enumerate(tqdm(srt_list[:], 'video clip by csv file start')):
            audio_output = f'{role_audios}/voice/{person}'
            os.makedirs(audio_output, exist_ok=True)
            index = str(index).zfill(4)
            text = make_filename_safe(subtitle)
            start_time = float(start_time)
            end_time = float(end_time)
            
            # ss = start_time.zfill(11).ljust(12, '0')[:12]
            # ee = end_time.zfill(11).ljust(12, '0')[:12]
            ss = srt_format_timestamp(start_time)
            ee = srt_format_timestamp(end_time)
            
            
            name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

            audio_output = f'{audio_output}/{name}.wav'
            ffmpeg_extract_audio(video_pth,audio_output,start_time,end_time)
            
def extract_pkl_feat(audio_extractor, role_audios):
    
    sub_dirs = get_subdir(f'{role_audios}/voice')
    
    for dir in sub_dirs[:]:
        # これはリストを返り値
        voice_files = get_filename(dir)
        name = os.path.basename(os.path.normpath(dir))
        for file, pth in tqdm(voice_files, f'extract {name} audio features ,convert .wav to .pkl'):
            new_dir = os.path.join(role_audios, 'embeddings', name)
            os.makedirs(new_dir, exist_ok=True)
            try:
                # サンプリングレートは16khzであるべき
                signal, fs = torchaudio.load(pth)
                # 録音の前処理
                signal_mono = torch.mean(signal, dim=0)
                # change freq
                resample_rate = 16000
                resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
                resampled_waveform = resampler(signal_mono)

                
                embeddings = audio_extractor.encode_batch(resampled_waveform)

                # 埋め込みを保存する
                with open(f"{new_dir}/{file}.pkl", "wb") as f:
                    pickle.dump(embeddings.detach().cpu(), f)
            except:
                continue
    
    log.info("録音データから埋め込みを作成しました。")
    

def load_model(model_name:str=None, device:str=None):
    if model_name == "speechbrain":
        try:
            from speechbrain.pretrained import EncoderClassifier
        except:
            "can't import speechbrain"
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                run_opts={"device": device},)
        return classifier
    
    elif model_name == "wavlm":
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        except:
            "can't import transformers"
            
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device)

        model_dict = {"feature_extractor": feature_extractor, 
                      "model": model}
        return model_dict
    else:
        raise "not included model"
        


class KNN_classifier:
    def __init__(self, audio_embds_dir, n_neighbors=3, 
                 threshold_certain=0.4, threshold_doubt=0.6) -> None:
        self.embeddings, self.labels = self.fetch_embeddings(audio_embds_dir)
        
        self.knn_cls = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # self.embeddings -> [batch_size, 1, hidden_dim]
        self.knn_cls.fit(self.embeddings.squeeze(1), self.labels)
        
        self.threshold_certain = threshold_certain
        self.threshold_doubt = threshold_doubt
    
    
    def fetch_embeddings(self, audio_embds_dir:str=None) -> list[np.ndarray, list[str]]:
        """ From a directory with folders named and labels and 
        inside the embeddings files, get the embeddings and the labels.

        Args:
            audio_embds_dir (str, optional): _description_. Defaults to None.

        Returns:
            _type_: 
        """
        embeddings_cls = None
        labels = []
        dim = 0
        
        # これはサブフォルダの名前をリストに格納する
        role_dirs = []
        for item in os.listdir(audio_embds_dir):
            if os.path.isdir(os.path.join(audio_embds_dir, item)):
                role_dirs.append(item)

        # キャラごとに埋め込みを読みます
        for role_dir in role_dirs:
            role = os.path.base(os.path.normpath(role_dir))
            # これは名前だけ、パずじゃない
            files_names = os.listdir(os.path.join(audio_embds_dir, role_dir))
            file_list = [os.path.join(audio_embds_dir, role_dir, embeddings_path) for embeddings_path in files_names]
            
            for embeddings_path in file_list:
                # 埋め込みファイルを開く
                with open(embeddings_path, 'rb') as fp:
                    embedding = pickle.load(fp)
                fp.close()
                
                # 前作ったリストに格納する
                if dim == 0:
                    embeddings_cls = embedding
                    dim = embeddings_cls.shape[0]
                else:
                    # This is equivalent to concatenation along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N)
                    embeddings_cls = np.vstack((embeddings_cls, embedding))

                labels.append(role)
                
        return embeddings_cls, labels

    def predict_class(self, embedding: torch.Tensor) -> list[str, float]:
        """_summary_

        Args:
            embedding (torch.Tensor): _description_

        Returns:
            list[str, float]: _description_
        """
        predicted_label = self.knn_cls.predict(embedding)
        dist, _ = self.knn_cls.kneighbors(embedding)
        # 一番近いクラスターの距離をとる
        dist = dist[0].min()
        
        # もしラベルがないなら、''を変える
        name = ''
        if dist < self.threshold_certain:
            name = predicted_label[0]
        elif dist < self.threshold_doubt:
            name = '（可能）' + predicted_label[0]
        
        return name, dist

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
            
            # ss = start_time.zfill(11).ljust(12, '0')[:12]
            # ee = end_time.zfill(11).ljust(12, '0')[:12]
            ss = srt_format_timestamp(start_time)
            ee = srt_format_timestamp(end_time)
            
            
            name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

            audio_output = f'{audio_output}/{name}.wav'
            ffmpeg_extract_audio(video_path,audio_output,start_time,end_time)
            
    # def ffmpeg_extract_audio(self, video_input, audio_output, start_time, end_time):
 
    #     command = ['ffmpeg', '-ss',str(start_time), '-to', str(end_time), '-i', f'{video_input}', "-vn",
    #                 '-c:a', 'pcm_s16le','-y', audio_output, '-loglevel', 'quiet']
        
    #     subprocess.run(command)
        

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
            
            for file, pth in tqdm(voice_files, f'extract {name} audio embeddings ,convert .wav to .pkl'):
                new_dir = os.path.join(save_folder, 'embeddings', name)
                os.makedirs(new_dir, exist_ok=True)
                try:
                    resampled_waveform = self.preprocess_audio(pth)
                    if self.model_type == "wavlm":
                        inputs = self.classifier["feature_extractor"](resampled_waveform, padding=True, return_tensors="pt", sampling_rate=16000)
                        inputs = inputs.to(self.device)
                        embeddings = self.classifier["model"](**inputs).embeddings
                        embeddings = F.normalize(embeddings.squeeze(1), p=2, dim=1)
                    else:  
                        embeddings = self.classifier.encode_batch(resampled_waveform)
                    
                    # 埋め込みを保存する
                    with open(f"{new_dir}/{file}.pkl", "wb") as f:
                        pickle.dump(embeddings.detach().cpu(), f)
                except Exception as e:
                    log.warning(f"Error when saving the embeddings. {e}")
                    continue
        log.info("録音データから埋め込みを作成しました。")
        
        
    def preprocess_audio(self, audio_path:str=None) -> torch.Tensor:
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
            
            # ss = start_time.zfill(11).ljust(12, '0')[:12]
            # ee = end_time.zfill(11).ljust(12, '0')[:12]
            ss = srt_format_timestamp(start_time)
            ee = srt_format_timestamp(end_time)
            
            
            name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

            audio_output = f'{temp_folder}/{filename}/{self.voice_dir}/{name}.wav'
            ffmpeg_extract_audio(video_path,audio_output,start_time,end_time)
            
    def extract_audios_for_labeling(self, annotate_csv:str=None, video_path:str=None, 
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
            
            # ss = start_time.zfill(11).ljust(12, '0')[:12]
            # ee = end_time.zfill(11).ljust(12, '0')[:12]
            ss = srt_format_timestamp(start_time)
            ee = srt_format_timestamp(end_time)
            
            
            name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

            audio_output = f'{temp_folder}/{filename}/{self.voice_dir}/{name}.wav'
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
        assert len(file_names) == len(df), log.error("The lenght of file_names is not enough")
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
            # print(pth)
            try:
                resampled_waveform = self.preprocess_audio(pth)
                    
                embeddings = self.classifier.encode_batch(resampled_waveform)


                # 埋め込みを保存する
                with open(f"{new_dir}/{file}.pkl", "wb") as f:
                    pickle.dump(embeddings.detach().cpu(), f)
            except:
                continue

        log.info("録音データから埋め込みを作成しました。")









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
    
   
        
    # data = pd.read_csv(annotation_file)
    # speechbrainのモデルを読み込む
    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
    #                                             run_opts={"device": "cuda"},)
    # video_pth = args.video_path
    
    # clip_audio_bycsv(args.annotate_map, video_pth, args.role_audios)
    
    # extract_pkl_feat(classifier, args.role_audios)
    
    classifier = load_model(model, device)
    processor = data_processor(classifier, model, device)
    # 録音データを格納する
    processor.extract_audios_by_csv(annotation_file, video_path, output_path)
    
    # 埋め込みを生成する
    processor.extract_embeddings(output_path)
    
    
def prepare_labeling(annotation_file:str=None,
         save_folder:str="tmp",
         video_path:str=None,
         ):
    
    processor = data_processor(None)
    # # 録音データを格納する
    processor.extract_audios_for_labeling(annotation_file, video_path, save_folder, iscropping=True)
    
    # TODO: add a function like the predict_2_csv of the predict.py that add the column with the file paths

    
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