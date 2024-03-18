import os

from .crop import crop
from ..common import log
from .predict import recognize
from .crop import data_processor
import asyncio

class Finder:
    model_opts = ["speechbrain", "wavlm"]
    
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
            

        return "Completado"
    
    
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
        
        
        return "Representaciones de personajes creadas!"
        
    async def make_predictions(self,
                    model:str=None,
                    device:bool=None,) -> str:
        """Predicts the character that said each line in the subtitles

        Args:
            model (str, optional): _description_. Defaults to None.
            device (bool, optional): _description_. Defaults to None.

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
        
        if device == True:
            device = "cuda"
        else:
            device = "cpu"   
        
        recognize(annotation_file=self.annotation_file,
        output_path=self.output_path,
        video_path=self.video_path,
        character_folder=self.character_folder,
        model=model,
        device=device,)
        
        
        return "Creadas predicciones!"
    
        
        
    def update_video_path(self, video_path:str=None):
        if video_path != None:
            self.video_path = os.path.join(self.input_path, video_path)
        
        
    def update_output_path(self, output_path:str=None):
        if output_path != None:
            self.output_path = output_path

    def update_annotation_file(self, annotation_file:str=None):
        if annotation_file != None:
            # it is already the whole path
            self.annotation_file = annotation_file
            log.info(f"Annotation file of finder updated to {self.annotation_file}")
            
