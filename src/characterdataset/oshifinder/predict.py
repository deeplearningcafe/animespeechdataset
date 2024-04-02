from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse


from .crop import data_processor, load_model
from ..common import log

class KNN_classifier:
    def __init__(self, audio_embds_dir, n_neighbors=3, 
                 threshold_certain=0.4, threshold_doubt=0.6) -> None:
        """This class is used for prediction, using k-means we classify new samples by comparing the distance with its nearest cluster

        Args:
            audio_embds_dir (str): path of the directory where the character embeddings are stored.
            n_neighbors (int, optional): number of clusters to use. Defaults to 3.
            threshold_certain (float, optional): maximum distance to consider the character. Defaults to 0.4.
            threshold_doubt (float, optional): maximum distance to consider the character as possible. Defaults to 0.6.
        """
        self.embeddings, self.labels = self.fetch_embeddings(audio_embds_dir)
        
        self.knn_cls = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        
        # self.embeddings -> [batch_size, 1, hidden_dim]
        # in the case of espnet -> [batch_size, hidden_dim]
        # self.knn_cls.fit(self.embeddings.squeeze(1), self.labels)
        self.knn_cls.fit(self.embeddings, self.labels)
        
        self.threshold_certain = threshold_certain
        self.threshold_doubt = threshold_doubt
    
    
    def fetch_embeddings(self, audio_embds_dir:str=None) -> list[np.ndarray, list[str]]:
        """ From a directory with folders named and labels and 
        inside the embeddings files, get the embeddings and the labels.

        Args:
            audio_embds_dir (str, optional): the path to the embeddings folder, which has subfolders named as their characters. Defaults to None.

        Returns:
            _type_: 
        """
        embeddings_cls = None
        labels = []
        dim = 0

        # add the embeddings folder
        audio_embds_dir = f"{audio_embds_dir}/embeddings"
        
        # これはサブフォルダの名前をリストに格納する
        role_dirs = []
        for item in os.listdir(audio_embds_dir):
            if os.path.isdir(os.path.join(audio_embds_dir, item)):
                role_dirs.append(item)

        

        # キャラごとに埋め込みを読みます
        for role_dir in role_dirs:
            log.info(f'{audio_embds_dir}/{role_dir}')
            
            role = os.path.basename(os.path.normpath(role_dir))
            # これは名前だけ、パずじゃない
            files_names = os.listdir(os.path.join(audio_embds_dir, role_dir))
            file_list = [os.path.join(audio_embds_dir, role_dir, embeddings_path) for embeddings_path in files_names]
            
            for embeddings_path in file_list:
                # 埋め込みファイルを開く
                with open(embeddings_path, 'rb') as fp:
                    embedding = pickle.load(fp)
                fp.close()
                # print(embedding.shape) torch.Size([1, 1, 192])
                # 前作ったリストに格納する
                if dim == 0:
                    embeddings_cls = embedding
                    dim = embeddings_cls.shape[0]
                else:
                    # This is equivalent to concatenation along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N)
                    embeddings_cls = np.vstack((embeddings_cls, embedding))

                labels.append(role)
        # print(embeddings_cls.shape) (239, 1, 192)
        return embeddings_cls, labels

    def predict_class(self, embedding: torch.Tensor) -> list[str, float]:
        """Given the embedding of the new sample, predict the class by comparing the distance with the labeled data

        Args:
            embedding (torch.Tensor): embedding from the new sample

        Returns:
            tuple[str, float]: returns the label and the distance to the nearest cluster
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
    
    def predict_2_csv(self, temp_folder:str=None, video_path:str=None, keep_unclassed:bool=False) -> None:
        """This function predicts the character of the cropped audios using KNN trained with the embeddings of the characters

        Args:
            temp_folder (str, optional): Folder where the audios and embeddings we want to predict are stored. Defaults to None.
            video_path (str, optional): path of the video to get the filename . Defaults to None.
            keep_unclassed (bool, optional): if true keep the predictions without class, that is that they are not labelled as any character. Defaults to False.
        """
        file = os.path.basename(video_path)
        filename, format = os.path.splitext(file)
        
        temp_dir = f'{temp_folder}/{filename}'
        temp_embeds = f'{temp_dir}/embeddings'

        
        # files names
        file_names = os.listdir(os.path.join(temp_embeds))
        file_list = [os.path.join(temp_embeds, embeddings_path) for embeddings_path in file_names]

        keep_index = []
        preds = []
        distances = []
        for i, path in enumerate(tqdm(file_list, f'predict label from audio embeddings')):
            with open(path, 'rb') as fp:
                embedding = pickle.load(fp)
            fp.close()
            embedding = embedding.squeeze(0)
            
            name, dist = self.predict_class(embedding)
            if not keep_unclassed:
                if len(name) > 0:
                    preds.append(name)
                    distances.append(dist)
                    keep_index.append(i)
                    
            else:
                preds.append(name)
                distances.append(dist)
        
        if not keep_unclassed:
            file_list = [file_list[i] for i in keep_index]
        
        assert ((len(file_list) == len(preds)) and (len(file_list) == len(distances))), "The lengths of the preds and the files list is not the same"
        # csvファイルに保存する
        # normalize path
        # csv_filename = f"{filename}_preds.csv"
        # csv_filename = f"preds.csv"
        csv_filename = f"{filename}.csv"
        
        # csv_filename = os.path.join(temp_dir, csv_filename)
        csv_filename =  f'{temp_dir}/{csv_filename}'
        csv_filename = os.path.normpath(csv_filename)
        df = pd.DataFrame({"filename": file_list, "predicted_label": preds, "distance": distances})
        
        
        
        df.to_csv(csv_filename, index=False)
        log.info(f"csvを保存しました{csv_filename}！")
        



def recognize_cmd(args):
    # checking if input_video is a file
    if not os.path.isfile(args.video_path):
        log.info('input_video is not exist')
        return
    
    # checking if input_srt is a file
    if not os.path.isfile(args.annotate_map):
        log.info('annotate_map is not exist')
        return
    
    # checking if role_audios is a folder
    if not os.path.isdir(args.save_folder):
        log.info('role_audios is not exist')
        return

    # checking if output_folder is a folder
    # if not os.path.isdir(args.output_path):
    #     print('warning output_path is not exist')
        # create output_folder
        # os.mkdir(args.output_folder)
        # print('create folder', args.output_folder)
        
    
    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
    #                                             run_opts={"device": "cuda"},)
    classifier = load_model(args.model, args.device)
    
    processor = data_processor(classifier)
    # # 録音データを格納する
    processor.extract_audios_by_subs(args.annotate_map, args.video_path, args.save_folder)
    
    # # 埋め込みを生成する
    processor.extract_embeddings_new(args.video_path, args.save_folder)
    
    knn = KNN_classifier(args.character_folder, n_neighbors=4)
    knn.predict_2_csv(args.save_folder, args.video_path, False)

    # delete the temp folder
    # shutil.rmtree(temp_folder)

def recognize(annotation_file:str=None,
         output_path:str=None,
         video_path:str=None,
         character_folder:str=None,
         n_neighbors:int=4,
         model:str=None,
         device:str=None,
         keep_unclassed:bool=None) -> None:
    """Predicts the character that said each line in the subtitles

    Args:
        annotation_file (str, optional): _description_. Defaults to None.
        output_path (str, optional): _description_. Defaults to None.
        video_path (str, optional): _description_. Defaults to None.
        character_folder (str, optional): _description_. Defaults to None.
        model (str, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to None.
    """
    log.info(f"Creating representations from annotations {annotation_file} "
             f"and video {video_path} with output at {output_path} "
             f"using character folder at {character_folder} "
             f"with n_neighbors {n_neighbors}, {model} and {device} "
             f"and keeping unclassed as {keep_unclassed}")
    classifier = load_model(model, device)
    
    processor = data_processor(classifier)
    # # 録音データを格納する
    processor.extract_audios_by_subs(annotation_file, video_path, output_path)
    log.info("Audios extracted")
    
    # # 埋め込みを生成する
    processor.extract_embeddings_new(video_path, output_path)
    log.info("Embeddings created")
    
    knn = KNN_classifier(character_folder, n_neighbors=n_neighbors)
    log.info(f"Starting predictions with output at {output_path}")
    knn.predict_2_csv(output_path, video_path, keep_unclassed=keep_unclassed)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='動画とcsvファイルから埋め込みと録音データを取得する'
    )
    # parser.add_argument("verbose", type=bool, action="store")
    parser.add_argument('--annotate_map', default='dataset_recognize.csv', 
                        type=str, required=True, help='せりふのタイミングとキャラ')
    parser.add_argument('--save_folder', default='./tmp', type=str,
                        required=True, help='出力を保存するためのフォルダ')
    parser.add_argument('--video_path', default=None, type=str, required=True,
                        help='録音データを取得するための動画')
    # parser.add_argument('--output_path', default=None, type=str, required=True,
    #                     help='録音データを取得するための動画')
    parser.add_argument('--character_folder', default=None, type=str, required=True,
                        help='事前にキャラと埋め込みのフォルダ')
    
    parser.add_argument('--model', default="speechbrain", type=str, required=True, choices=["speechbrain", "wavlm"],
                        help='埋め込みを作成するためのモデル')
    parser.add_argument('--device', default="cuda", type=str, required=False, choices=["cuda", "cpu"],
                    help='埋め込みを作成するためのモデル')
    
    args = parser.parse_args()
    parser.print_help()
    # print(args)
    recognize_cmd(args)