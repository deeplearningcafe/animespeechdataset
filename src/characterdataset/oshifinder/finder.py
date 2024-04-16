import os

from .crop import crop
from ..common import log
from .predict import recognize
from .crop import data_processor
import asyncio
import pandas as pd
import shutil
from tqdm import tqdm

class Finder:
    model_opts = ["speechbrain", "wavlm", "espnet"]
    
    def __init__(self,
                 input_path:str=None,
                 annotation_file:str=None,
                 output_path:str=None,
                 video_path:str=None,
                 output_path_labeling:str=None,
                 character_folder:str=None,
            ) -> None:
        self.input_path = input_path
        self.annotation_file = annotation_file
        self.output_path = output_path
        self.video_path = video_path
        self.output_path_labeling = output_path_labeling
        self.character_folder = character_folder

    def crop_for_labeling(self, annotation_file:str=None) -> str:
        """Given the csv file, str converted, it adds a new column with the path of the created audios.

        Args:
            annotation_file (str, optional): path to the annotations file(character, start, end, text). Defaults to None.

        Returns:
            str: returns a message with the result, completed or error
        """
        # check if annotate_map is a file
        self.update_annotation_file(annotation_file)
        if not os.path.isfile(self.annotation_file):
            raise ValueError(
                    f"The annotation file at {self.annotation_file} does not exists"
                )
        # checking if input_video is a file
        if self.video_path is None:
                raise ValueError(
                    f"Provide a video file"
                )
        if not os.path.isfile(self.video_path):
            raise ValueError(
                    f"The video file at {self.video_path} does not exists"
                )

        # if not os.path.isdir(self.output_path_labeling):
        # log.info(f'temp folder to save clips {self.output_path_labeling} does not exist')
        # no need to log this part, as it is a temp folder
        os.makedirs(self.output_path_labeling, exist_ok=True)
           
        data_processor.extract_audios_for_labeling(
                    annotate_csv=self.annotation_file,
                    temp_folder=self.output_path_labeling,
                    video_path=self.video_path, 
                    iscropping=True)
            

        return "Success"
    
    
    def crop_files(self, 
                    model:str=None,
                    device:bool=None,
                    ) -> str:
        """Creates embeddings and save them in each character folder. This data will be used for predicting as the labeled data.
        First extract audios and then extract embeddings from the audios.

        Args:
            model (str, optional): name of the model to use for extracting the embeddings. Defaults to None.
            device (str, optional): device to use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            str: sucess if completed
        """
    
        # check if annotate_map is a file
        if self.annotation_file is None:
                raise ValueError(
                    f"Provide an annotation file"
                )
        if not os.path.isfile(self.annotation_file):
            raise ValueError(
                    f"The annotation file at {self.annotation_file} does not exists"
                )
        # checking if input_video is a file
        if self.video_path is None:
                raise ValueError(
                    f"Provide a video file"
                )
        if not os.path.isfile(self.video_path):
            raise ValueError(
                    f"The video file at {self.video_path} does not exists"
                )

        # check if embeddings are a folder
        if not os.path.isdir(self.character_folder):
            log.info(f'character embeddings folder {self.character_folder} does not exist')
            # create role_audios folder
            os.makedirs(self.character_folder)
            
        if device == True:
            device = "cuda"
        else:
            device = "cpu"
           
        crop(annotation_file=self.annotation_file,
            output_path=self.character_folder,
            video_path=self.video_path,
            model=model,
            device=device,
            )
        
        
        return "Characters embeddings have been created!"
        
    async def make_predictions(self,
                    n_neighbors:int=4,
                    model:str=None,
                    device:bool=True,
                    keep_unclassed:bool=False) -> str:
        """Predicts the character that said each line in the subtitles

        Args:
            model (str, optional): _description_. Defaults to None.
            device (bool, optional): _description_. Defaults to True.
            keep_unclassed (bool, optional): _description_. Defaults to False.
        
        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            str: sucess if completed
        """
        
        # as we need the annotation file to be updated, we need to wait to avoid none in the annotation_file
        await asyncio.sleep(0.2)
        
        # Check the inputs
        log.info("Starting predictions")
        # check if annotate_map is a file
        if self.annotation_file is None:
                raise ValueError(
                    f"Provide an annotation file"
                )
        if not os.path.isfile(self.annotation_file):
            raise ValueError(
                    f"The annotation file at {self.annotation_file} does not exists"
                )
        # checking if input_video is a file
        if self.video_path is None:
                raise ValueError(
                    f"Provide a video file"
                )
        if not os.path.isfile(self.video_path):
            raise ValueError(
                    f"The video file at {self.video_path} does not exists"
                )

        # check if role_audios is a folder
        if not os.path.isdir(self.output_path):
            log.info(f'output folder {self.output_path} does not exist')
            # create role_audios folder
            os.mkdir(self.output_path)
        
        # check if role_audios is a folder
        if not os.path.isdir(self.character_folder):
            raise ValueError(
                    f"The folder with embeddings at {self.character_folder} does not exists"
                )
            
        n_neighbors = int(n_neighbors)
        if n_neighbors < 2:
            log.warning(f"The k {n_neighbors} can't be used, using 4 instead")
            n_neighbors = 4
        
        if device == True:
            device = "cuda"
        else:
            device = "cpu"   
        
        # the check of the model parameter is done inside the load_model function of predict.py
        output_filename = recognize(annotation_file=self.annotation_file,
            output_path=self.output_path,
            video_path=self.video_path,
            character_folder=self.character_folder,
            n_neighbors=n_neighbors,
            model=model,
            device=device,
            keep_unclassed=keep_unclassed)
        
        
        return "Predictions have been completed!", output_filename
    
    
    def add_character_embeddings(self, predictions_path:str, minimum_distance:float=0.2) -> tuple[str, str]:
        """From the cleaned prediction file, copies the embeddings to their character folders.
        It takes the samples with distance higher than a threshold, as the predictions should have been already
        cleaned, we are adding informative samples to the labeled data which should improve predictions accuracy.
        Args:
            predictions_path (str): _description_
            minimum_distance (float): use embeddings with distance higher than this th. Defaults to 0.2

        Returns:
            tuple[str, str]: _description_
        """
        
        predictions_path = f"{self.output_path}/{predictions_path}"
        # dfを読み込む
        df = pd.read_csv(predictions_path, header=0)
        # キャラのいないサンプルを排除します
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        # add the embeddings folder
        character_embeddings = f"{self.character_folder}/embeddings"
        
        # これはサブフォルダの名前をリストに格納する
        character_names = []
        for item in os.listdir(character_embeddings):
            if os.path.isdir(os.path.join(character_embeddings, item)):
                character_names.append(item)
        log.info(f"The characters embeddings are {character_names}")

        
        for i in tqdm(range(len(df)), "generate text from title"):
            distance = df.iloc[i, 2]
            
            if distance > minimum_distance:
                embedding_path = df.iloc[i, 0]
                # this is the embeddings .pkl
                filename = os.path.basename(embedding_path)
                
                predicted_label = df.iloc[i, 1]
                # （可能）は既にいないはずけど、もしあったら使えないようにします
                for character in character_names:
                    if predicted_label == character:
                        save_path = f'{character_embeddings}/{character}/{filename}'
                        log.info(f"Adding file {filename} to {character} folder")
                        # should include error handling
                        shutil.copy2(embedding_path, save_path)
                        continue
        
        return "Copied embedding files to the character embeddings folders"
        
        
    def update_video_path(self, video_path:str=None):
        if video_path != None:
            # self.video_path = os.path.join(self.input_path, video_path)
            self.video_path = f"{self.input_path}/{video_path}"

        
        
    def update_output_path(self, output_path:str=None):
        if output_path != None:
            self.output_path = output_path

    def update_annotation_file(self, annotation_file:str=None):
        if annotation_file != None:
            # it is already the whole path
            self.annotation_file = annotation_file
            log.info(f"Annotation file of finder updated to {self.annotation_file}")
            
