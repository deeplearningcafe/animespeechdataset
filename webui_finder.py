import os
import gradio as gr
import pandas as pd
import shutil
from characterdataset.common import log
from characterdataset.configs import load_global_config

from characterdataset.datasetmanager import DatasetManager
from characterdataset.oshifinder import Finder


# load config
config = load_global_config()


dataset_manager = DatasetManager(
    dataset_type=config.dataset_manager.dataset_type,
    input_path=config.dataset_manager.input_path,
    output_path=config.dataset_manager.output_path,
    num_characters=config.dataset_manager.num_characters,
    time_interval=config.dataset_manager.time_interval,

    )

finder = Finder(
    input_path=config.dataset_manager.input_path,
    output_path=config.dataset_manager.output_path,
    output_path_labeling=config.finder.temp_folder,
    character_folder=config.finder.character_embedds,
)

# create folders
if not os.path.isdir(config.dataset_manager.input_path):
    log.info('Creating input folders')
    # create output_folder
    os.makedirs(config.dataset_manager.input_path, exist_ok=True)
if not os.path.isdir("pretrained_models"):
    log.info('Creating folder for storing models')
    # create output_folder
    os.makedirs("pretrained_models", exist_ok=True)

if not os.path.isdir(config.dataset_manager.output_path):
    log.info('Creating output folders')
    # create output_folder
    os.makedirs(config.dataset_manager.output_path, exist_ok=True)



def call_function(dataset_type:str, transcribe:bool=False, ) -> str:
    log.info(f"Calling function of {dataset_type}")
    
    if not transcribe:
        if dataset_type == DatasetManager.dataset_types[0]:
            result = dataset_manager.create_dialogues()
        elif dataset_type == DatasetManager.dataset_types[1]:
            result = dataset_manager.create_audio_files()

        else:
            return "Función desconocida"
    else:
        log.info("Transcribing video")
        result = dataset_manager.transcribe_video(video_path=finder.video_path, 
                                                  iscropping=False)

    return result  
    

def update_visibility(dataset_type:str) -> tuple[gr.Column, gr.Column, gr.Row]:
    """Changes the visibility of the different parts of the export dataset based on the type of 
    dataset to export

    Args:
        dataset_type (str): type of dataset to export

    Returns:
        tuple[gr.Column, gr.Column, gr.Row]: changed visibility of components
    """
    if dataset_type == DatasetManager.dataset_types[0]:
        return gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=True)
    elif dataset_type == DatasetManager.dataset_types[1]:
        return gr.Column(visible=False), gr.Column(visible=True), gr.Row(visible=False)

def load_df(csv_path: str=None) -> None:
    """Given a csv path, it reads it and creates a dataframe

    Args:
        csv_path (str, optional): _description_. Defaults to None.
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.drop(['start_time', 'end_time'], axis=1)
    return df

def load_audio(audio_path: str=None, audio_component: gr.Audio=None) -> gr.Audio:
    """Updates the value of an audio element

    Args:
        audio_path (str, optional): _description_. Defaults to None.
        audio_data (gr.Audio, optional): _description_. Defaults to None.

    Returns:
        gr.Audio: _description_
    """
    audio_component = gr.Audio(value=audio_path)
    return audio_component

def save_df(df: pd.DataFrame=None, csv_path: str=None) -> str:
    """Saves the new df and returns the new path. Also remove the folder created for the clipped audios.

    Args:
        csv_path (str, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    *filename, format = os.path.splitext(csv_path)
    filename = f'{"".join(filename)}_updated.csv'
    log.info(filename)
    
    

    # remove the rows without annotation
    df_original = pd.read_csv(csv_path, header=0)
    for i in range(len(df)):
        if len(df.iloc[i, 0]) > 0:
            df_original.iloc[i, 0] = df.iloc[i, 0]
    
    # remove the filename column
    df_original = df_original.drop("filename", axis=1)
    df_original = df_original.dropna()
    df_original.to_csv(filename, index=False)
    log.info(f'CSVファイル "{filename}" にデータを保存しました。')
    
    # delete the temp folder
    try:
        shutil.rmtree(finder.output_path_labeling)
        log.info(f"Deleted the audios folder at {finder.output_path_labeling}")
    except Exception as e:
        # we want to continue despite not removing this folder
        log.error(f"Could not remove the temp folder {e}")
        pass
        
    return filename, "Database have been updated!"
    
def create_labeling_data(transcribe:bool=False) -> tuple:
    """It calls 3 function, the inputs should be already updated in their classes.
    First transforms .str file to csv file, then clips audios for labeling and finally re

    Returns:
        tuple: (result of the function, the path of the file for annotations, the dataframe from that file)
    """
    
    # 1. Transform subs
    result = "Error"

    if transcribe:
        log.info("Transcribing video")
        result, annotation_file = dataset_manager.transcribe_video(video_path=finder.video_path,
                                                                   iscropping=True)
    else:
        result, annotation_file = dataset_manager.create_csv(crop=True)
    

    # 2. Crop audios, the filename does not change
    result = finder.crop_for_labeling(annotation_file)

    # 3. Load the csv file
    df = load_df(annotation_file)
    
    return result, annotation_file, df

def csv_for_predictions() -> str:
    result = dataset_manager.create_csv(crop=False)
    return result



GRADIO_THEME: str = "NoCrypt/miku"

initial_msg = """
# AnimeSpeech: Dataset Generation for Language Model Training and Text-to-Speech Synthesis from Anime Subtitles
From animes extract the dialogs between desired characters for training LLM, and extract the audios of one character to train TTS.

## Inputs
- Video file: the video from which take the audios and the dialogs.
- Subtitles file: the .str file containing the subtitles of the video.
Both of them should be placed at the data/inputs folder. In the textbox just include the name, the full path is not needed.

## Funcionalities
### Create annotations
Inputs should be the subtitles and the video.
- Character creation: user labels the data to create representations of the desired characters.
The converted subtitles become tabular data(like excel). 

- Character prediction: predict the character of each line. The desired character representations(embeddings) should be already created, 
it only predicts characters that have representations.

### Create datasets
Input should be the annotations file.
- Dialogues dataset: create conversational dataset for training LLM, user can choose the characters from whom take the dialogs.

- Audios dataset: extract all the audios of the desired character and create a folder with those and the text for TTS training.

## Directories form
```
├── data
│   ├── inputs
│   │   ├── subtitle-file.str
│   │   ├── video-file
│   ├── outputs
│   │   ├── subtitle-file.csv
│   │   ├── video-file
│   │   │   ├── preds.csv
│   │   │   ├── voice
│   │   │   ├── embeddings
├── pretrained_models
├── src
│   ├── characterdataset
│   │   ├── common
│   │   ├── configs
│   │   ├── datasetmanager
│   │   ├── oshifinder
├── tests
│   ├── test_dataset_manager.py
│   ├── test_finder.py
├── webui_finder.py
```
"""

def create_ui():

    with gr.Blocks(theme=GRADIO_THEME) as dataset_app:
        gr.Markdown(initial_msg)


        with gr.Row():
            with gr.Column():
                subtitles_file = gr.Textbox(
                                label="Subtitles file name.",
                                placeholder="Name-of-your-file",
                                info="Insert the name of the str file, which is in the folder data/inputs.",
                            )
            with gr.Column():
                video_path = gr.Textbox(
                                label="Video file name.",
                                placeholder="Name-of-your-file",
                                info="Insert the name of the video file, which is in the folder data/inputs.",
                            )
                transcribe = gr.Checkbox(label="Transcribe", info="In the case of not having the subtitle, from the video create the annotations.",
                    value=False)
        with gr.Row():
            with gr.Accordion("Advanced options", open=False):
                with gr.Column():    
                    num_characters = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=1,
                                label="Minimum number of characters to use the phrase.",
                                interactive=True
                            )
                with gr.Column():
                    model = gr.Dropdown(
                        choices=Finder.model_opts,
                        label="Type of the model to extract the embeddings.",
                        value=Finder.model_opts[0]
                    )
                with gr.Row():
                    device = gr.Checkbox(
                        label="cuda", info="In case to use GPU.(Recommended)", value=True
                    )


        with gr.Row():
            annotation_file = gr.Textbox(label="Annotation file name.",
                                        info="Only use for dialogues and audio datasets.", visible=True)
            result = gr.Textbox(label="Result")
            
        # For the crop given the subs
        with gr.Tab("Create characters"):
                        
            # For the audios
            with gr.Column(visible=True) as labeling:
                with gr.Accordion(label="Create dataset", open=False):
                    with gr.Row():
                        with gr.Column(min_width=640):
                            load_data = gr.Button("Create dataset for labeling")
                            dataframe = gr.DataFrame(interactive=True, row_count=100)
                            
                    
                    with gr.Row():        
                        with gr.Column():
                            path_audio = gr.Textbox(label="Name of the audio file", info="You can copy it from the dataframe")
                            audio_button = gr.Button("Load the audio")
                        with gr.Column():
                            audio_data = gr.Audio(
                                    label="Audio",
                                )
                                            
                    with gr.Row():
                        save_button = gr.Button("Save annotations")    
                    

            

                with gr.Column(visible=True) as crop:
                    with gr.Accordion(label="Create representations", open=False):
                        with gr.Row():
                            embedds_button = gr.Button("Create representations(embeddings) of the characters")
                

        # For the predict given the subs
        with gr.Tab("Predict characters"):
            with gr.Row():
                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        keep_unclassed = gr.Checkbox(
                            label="Keep other characters", info="In case to keep lines of characters others than the desired characters", value=False
                        )
                        n_neighbors = gr.Slider(
                            label="Number of neighbors for KNN",
                            info="Number of neighbors to use, should be at least the same number of characters to predict.",
                            minimum=2,
                            maximum=10,
                            value=4,
                            step=1,
                        )

            with gr.Column(visible=True) as predict:                        
                predict_button = gr.Button("Predict")
        
        # For creating dialogs and audio datasets
        with gr.Tab("Export for training"):
            # select the function we want to use
            dataset_type = gr.Dropdown(
                choices=DatasetManager.dataset_types,
                label="Type of dataset to process",
                value=dataset_manager.dataset_type
            )
            # For the dialogs
            with gr.Column(visible=True) as dialogs:
                with gr.Row():
                    with gr.Column():
                        first_character = gr.Textbox(
                                label="First character, user role.",
                                placeholder="Name-of-your-character",
                                info="Name of the character that makes the questions, in the prompt the user role.",
                            )
                    with gr.Column():
                        second_character = gr.Textbox(
                                label="Second character, system role.",
                                placeholder="Name-of-your-character",
                                info="Name of the character that makes the answers, in the prompt the system role.",
                            )
            # For the audios
            with gr.Column(visible=False) as audios:
                with gr.Row():
                    with gr.Column():
                        character = gr.Textbox(
                                label="Character to create audios",
                                placeholder="Name-of-your-character",
                                info="Name of the character to create the audio dataset.",
                            )
            
            with gr.Row() as advaced_export:
                with gr.Accordion("Opciones avanzadas", open=False):
                    with gr.Column():
                        time_interval = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Maximum interval between dialogue lines, seconds",
                            interactive=True
                        )

            with gr.Row():
                transcribe_button = gr.Button("Transform")

        # this is for the dialgs and audios
        transcribe_button.click(
            call_function,
            inputs=[
                dataset_type, transcribe
            ],
            outputs=[result],
        )
        
        # we will update the values of the dataset_manager class when the elements change
        dataset_type.change(
            dataset_manager.update_dataset_type,
            inputs=[dataset_type],   
        ).then(update_visibility, inputs=[dataset_type], outputs=[dialogs, audios, advaced_export])

        
        subtitles_file.change(
            dataset_manager.update_subtitles_file,
            inputs=[subtitles_file],   
        )
        
        annotation_file.change(
            finder.update_annotation_file,
            inputs=[annotation_file],   
        ).then(dataset_manager.update_annotation_file,
            inputs=[annotation_file],
            )
        
        video_path.change(
            finder.update_video_path,
            inputs=[video_path],   
        )

        load_data.click(
            create_labeling_data, inputs=[transcribe],
            outputs=[result, annotation_file, dataframe]
        )
        audio_button.click(
            load_audio,
            inputs=[path_audio, audio_data], outputs=[audio_data]
        )
        
        save_button.click(
            save_df,
            inputs=[dataframe, annotation_file], outputs=[annotation_file, result]
        )
        num_characters.change(
            dataset_manager.update_num_characters,
            inputs=[num_characters],   
        )
        
        embedds_button.click(
            finder.crop_files,
            inputs=[model, device],
            outputs=[result]
        )
        
        predict_button.click(
            csv_for_predictions, outputs=[result, annotation_file]
        ).then(finder.make_predictions,
            inputs=[n_neighbors, model, device, keep_unclassed],outputs=[result])
        
        # audios and dialogs
        first_character.change(
            dataset_manager.update_first_character,
            inputs=[first_character],   
        )
        second_character.change(
            dataset_manager.update_second_character,
            inputs=[second_character],   
        )
        time_interval.change(
            dataset_manager.update_time_interval,
            inputs=[time_interval],   
        )
        character.change(
            dataset_manager.update_character,
            inputs=[character],   
        )
        
        
        
    return dataset_app

if __name__ == "__main__":

    webui_dataset = create_ui()
    webui_dataset.launch()