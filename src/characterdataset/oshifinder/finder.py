import os

from .crop import crop, prepare_labeling
from common.log import log
from .predict import recognize
from .crop import data_processor


class Finder:
    model_opts = ["speechbrain", "wavlm"]
    
    def __init__(self,
                 input_path:str=None,
                 annotation_file:str=None,
                 output_path:str=None,
                 video_path:str=None,
                 model:str=None,
                 device:str=None,
                 output_path_labeling:str=None,
                 character_folder:str=None,
            ) -> None:
        self.input_path = input_path
        self.annotation_file = annotation_file
        self.output_path = output_path
        self.video_path = video_path
        self.model = model
        self.device = device
        self.output_path_labeling = output_path_labeling
        self.character_folder = character_folder

    def crop_for_labeling(self, annotation_file:str=None) -> str:
        """Given the csv file, str converted, it adds a new column with the path of the created audios.

        Args:
            annotation_file (str, optional): path to the annotations file(character, start, end, text). Defaults to None.

        Returns:
            str: returns a message with the result, completed or error
        """
        # Check the inputs
        # output_path = os.path.join(self.output_path, output_path_labeling)
        # check if annotate_map is a file
        
        self.update_annotation_file(annotation_file)
        if not os.path.isfile(self.annotation_file):
            log.info(f'annotate_map {self.annotation_file} does not exist')
            return

        # check if role_audios is a folder
        if not os.path.isdir(self.output_path_labeling):
            log.info(f'temp folder to save clips {self.output_path_labeling} does not exist')
            # create role_audios folder
            os.mkdir(self.output_path_labeling)
           
        try:
            data_processor.extract_audios_for_labeling(
                    annotate_csv=self.annotation_file,
                    temp_folder=self.output_path_labeling,
                    video_path=self.video_path, 
                    iscropping=True)
            
            # prepare_labeling(annotation_file=self.annotation_file,
            # save_folder=self.output_path_labeling,
            # video_path=self.video_path,
            # )
        
        except Exception as e:
            log.warning(f"Error when cropping for labeling. {e}")
            return "Error"
        return "Completado"
    
    
    def crop_files(self, 
                #    output_path_labeling:str="tmp",
                    model:str=None,
                    device:str=None,
                    ) -> str:
        # Check the inputs
    
        # check if annotate_map is a file
        if not os.path.isfile(self.annotation_file):
            log.info(f'annotate_map {self.annotation_file} does not exist')
            return

        # check if role_audios is a folder
        # output_path = os.path.join(self.output_path, output_path_labeling)
        if not os.path.isdir(self.character_folder):
            log.info(f'character embeddings folder {self.character_folder} does not exist')
            # create role_audios folder
            os.mkdir(self.character_folder)
            
        if device == True:
            device = "cuda"
        else:
            device = "cpu"
           
        try: 
            crop(annotation_file=self.annotation_file,
            output_path=self.character_folder,
            video_path=self.video_path,
            model=model,
            device=device,
            )
        
        except Exception as e:
            log.warning(f"Error when cropping. {e}")
            return "Error"
        
        return "Representaciones de personajes creadas!"
        
    def make_predictions(self,
                    # character_folder:str="tmp",
                    model:str=None,
                    device:str=None,) -> str:
        # Check the inputs
        log.info("Starting predictions")
        # check if annotate_map is a file
        if not os.path.isfile(self.annotation_file):
            log.info(f'annotate_map {self.annotation_file} does not exist')
            return

        # check if role_audios is a folder
        if not os.path.isdir(self.output_path):
            log.info(f'output folder {self.output_path} does not exist')
            # create role_audios folder
            os.mkdir(self.output_path)
        
        # check if role_audios is a folder
        # character_folder = os.path.join(self.output_path, character_folder)
        if not os.path.isdir(self.character_folder):
            log.info(f'role_audios {self.character_folder} does not exist')

            return
        
        if device == True:
            device = "cuda"
        else:
            device = "cpu"   
        
        try: 
            recognize(annotation_file=self.annotation_file,
            output_path=self.output_path,
            video_path=self.video_path,
            character_folder=self.character_folder,
            model=model,
            device=device,)
        
        except Exception as e:
            log.warning(f"Error when predicting. {e}")
            return "Error"
        
        return "Completado"
        
    def update_video_path(self, video_path:str=None):
        if video_path != None:
            self.video_path = os.path.join(self.input_path, video_path)
        
    def update_model(self, model:str=None):
        if model != None:
            self.model = model
        
    def update_output_path(self, output_path:str=None):
        if output_path != None:
            self.output_path = output_path

    def update_annotation_file(self, annotation_file:str=None):
        if annotation_file != None:
            # self.annotation_file = os.path.join(self.output_path, annotation_file)
            # it is already the whole path
            self.annotation_file = annotation_file
            log.info(f"Annotation file of finder updated to {self.annotation_file}")
            
    def update_device(self, device:bool=True):
        if device == True:
            self.device = "cuda"
        else:
            self.device = "cpu"    
        
    # def update_output_path(self, output_path_labeling:str=None):
    #     if output_path_labeling != None:
    #         self.output_path_labeling = output_path_labeling