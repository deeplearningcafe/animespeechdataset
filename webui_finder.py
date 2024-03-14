import argparse
import os
import gradio as gr
import subprocess
import sys
import pandas as pd
import shutil


import logging
from common.log import log
from configs import load_global_config

from characterdataset.datasetmanager import DatasetManager
from characterdataset.oshifinder import Finder
# log_filename = "logs/dataset_manager.log"
# os.makedirs(os.path.dirname(log_filename), exist_ok=True)

# logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.DEBUG, format="%(asctime)s %(levelname)-7s %(message)s")

# log = logging.getLogger(__name__)
# python = sys.executable


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

def call_function(dataset_type:str, transcribe:bool=False, ) -> str:
    log.info(f"Calling function of {dataset_type}")
    
    if not transcribe:
        # if dataset_type == DatasetManager.dataset_types[0]:
        #     dataset_manager.update_crop(crop=True)
        #     result = dataset_manager.create_csv()
        if dataset_type == DatasetManager.dataset_types[0]:
            result = dataset_manager.create_dialogues()
        elif dataset_type == DatasetManager.dataset_types[1]:
            result = dataset_manager.create_audio_files()

        else:
            return "Función desconocida"
    else:
        log.info("Transcribing video")
        # this is probably really bad but maybe it works
        result = dataset_manager.transcribe_video(video_path=finder.video_path, 
                                                  iscropping=True)

    return result  
    

# def update_visibility(result:str) -> None:
#     if result == "Completado":
#         return gr.Column(visible=True)
def update_visibility(dataset_type:str) -> None:
    if dataset_type == DatasetManager.dataset_types[0]:
        return gr.Column(visible=True), gr.Column(visible=False)
    elif dataset_type == DatasetManager.dataset_types[1]:
        return gr.Column(visible=False), gr.Column(visible=True) 

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
    # if audio_component == None:
    audio_component = gr.Audio(value=audio_path)
    # else:
    #     audio_component.value = audio_path
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
        log.warning(f"Could not remove the temp folder {e}")
        
    return filename, "Base de datos actualizada!"
    
def create_labeling_data(transcribe:bool=False) -> tuple:
    """It calls 3 function, the inputs should be already updated in their classes.
    First transforms .str file to csv file, then clips audios for labeling and finally re

    Returns:
        tuple: (result of the function, the path of the file for annotations, the dataframe from that file)
    """
    
    # 1. Transform subs
    result = "Error"
    # result, annotation_file = call_function("subtitles", transcribe)
    # dataset_manager.update_crop(crop=True)
    if transcribe:
        log.info("Transcribing video")
        # this is probably really bad but maybe it works
        result, annotation_file = dataset_manager.transcribe_video(video_path=finder.video_path,
                                                                   iscropping=False)
    else:
        result, annotation_file = dataset_manager.create_csv(crop=True)
    

    # 2. Crop audios, the filename does not change
    result = finder.crop_for_labeling(annotation_file)

    # 3. Load the csv file
    df = load_df(annotation_file)
    
    return result, annotation_file, df

def csv_for_predictions() -> str:
    # dataset_manager.update_crop(False)
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

```
"""

def create_ui():

    with gr.Blocks(theme=GRADIO_THEME) as dataset_app:
        
        # select the function we want to use
        # dataset_type = gr.Dropdown(
        #     choices=DatasetManager.dataset_types,
        #     label="Tipo de archivo para procesar",
        #     value=dataset_manager.dataset_type
        # )
        gr.Markdown(initial_msg)


        with gr.Row():
            with gr.Column():
                subtitles_file = gr.Textbox(
                                label="Nombre del archivo de subtítulos.",
                                placeholder="Nombre-de-tu-archivo",
                                info="Inserte el nombre del archivo str, que está en la carpeta data/inputs",
                            )
            with gr.Column():
                video_path = gr.Textbox(
                                label="Nombre del archivo de video.",
                                placeholder="Nombre-de-tu-archivo",
                                info="Inserte el nombre del archivo de video, que está en la carpeta data/outputs",
                            )
                transcribe = gr.Checkbox(label="Transcribir", info="En el caso de no tener subtítulos, a partir del video se crean las anotaciones",
                    value=False)
        with gr.Row():
            with gr.Accordion("Opciones avanzadas", open=False):
                with gr.Column():
                        
                    # is_crop = gr.Checkbox(label="cropping", info="En el caso de usar el archivo para classificar personajes manualmente",
                    #                     value=False, visible=False)         
                    num_characters = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=1,
                                label="Mínimo número de caracteres para usar la frase.",
                                interactive=True
                            )
                with gr.Column():
                    model = gr.Dropdown(
                        choices=Finder.model_opts,
                        label="Tipo de modelo para crear representaciones del personaje.",
                        value=Finder.model_opts[0]
                    )
                with gr.Column():
                    device = gr.Checkbox(
                        label="cuda", info="En el caso de usar la gpu.(Recomendado)", value=True
                    )

        with gr.Row():
            # This file should not be visible
            annotation_file = gr.Textbox(label="Nombre del archivo de anotaciones.",
                                        info="Solo usar para crear audios y diálogos.", visible=True)
            result = gr.Textbox(label="Resultado")
            
        # For the crop given the subs
        with gr.Tab("Crear personajes"):
            # Base case
            # with gr.Accordion(label="Subtítulos", open=True):
            #     with gr.Row() as base:

                    # with gr.Row():
                    #     with gr.Column():
                    #         with gr.Row():
                    #             transcribe_button = gr.Button("Transformar")
                            # with gr.Row():
                            #     result_subs = gr.Textbox(label="Resultado")
                    
                # with gr.Row():
                #     annotation_file = gr.Textbox(label="Directorio del archivo con las anotaciones", visible=True)
                
            # with gr.Accordion(label="Anotaciones", open=False):
            #     with gr.Column(visible=True) as crop_labeling:
                    # with gr.Row():
                    #     with gr.Column():
                            
                            
                        # this is really not necessary, as we can use tmp here, this files should be deleted after completing the 
                        # annotations anyway.
                        # with gr.Column():
                            # path_labeling = gr.Textbox(
                            #     label="Nombre de la carpeta para guardar los clips.",
                            #     placeholder="Nombre-de-tu-carpeta",
                            #     info="Inserte el nombre de la carpeta, en la que guardar los audios",
                            # )
                            # annotation_file = gr.Textbox(
                            #     label="Nombre del archivo de video.",
                            #     placeholder="Nombre-de-tu-archivo",
                            #     info="Inserte el nombre del archivo de video, que está en la carpeta data/outputs",
                            # )
                    # with gr.Row():
                    #     with gr.Column():
                    #         with gr.Row():
                    #             with gr.Column():
                    #                 crop_label_button = gr.Button("Audios para anotaciones")
                                # with gr.Column():
                                #     result_crop_label = gr.Textbox(label="Resultado del crop para anotaciones")

                        
            # For the audios
            with gr.Column(visible=True) as labeling:
                with gr.Accordion(label="Crear dataset", open=False):
                    with gr.Row():
                    #     with gr.Column():
                    #         user_pred = gr.Textbox(
                    #                 label="Nombre de la carpeta de los audios.",
                    #                 placeholder="Nombre-de-tu-archivo",
                    #                 info="Nombre del personaje que hace las preguntas, en el prompt el rol de usuario",
                    #             )
                    #     with gr.Column():
                    #         audio_data = gr.Audio(
                    #                 label="Audio",
                    #             )
                    # with gr.Row():
                    #     with gr.Column():
                    #         prev_button = gr.Button("Anterior")
                    #     with gr.Column():
                    #         submit_button = gr.Button("Anotar")
                    #     with gr.Column():
                    #         post_button = gr.Button("Siguiente")
                    # with gr.Row():
                    #     delete_button = gr.Button("Borrar esta frase")
                    #     index = gr.Slider(
                    #                 minimum=0,
                    #                 maximum=50,
                    #                 value=0,
                    #                 step=1,
                    #                 label="Intervalo máximo entre diálogos, segundos",
                    #                 interactive=True
                    #             )
                        with gr.Column(min_width=640):
                            load_data = gr.Button("Crear base de datos")
                            dataframe = gr.DataFrame(interactive=True, row_count=100)
                            
                    
                    with gr.Row():        
                        with gr.Column():
                            path_audio = gr.Textbox(label="nombre del archivo de audio", info="puedes copiarlo del dataframe")
                            audio_button = gr.Button("Cargar el audio")
                        with gr.Column():
                            audio_data = gr.Audio(
                                    label="Audio",
                                )
                                            
                    with gr.Row():
                        save_button = gr.Button("Guardar anotaciones")    
                    

            

                with gr.Column(visible=True) as crop:
                    # with gr.Row():
                    #     with gr.Column():
                    #         video_path = gr.Textbox(
                    #             label="Nombre del archivo de video.",
                    #             placeholder="Nombre-de-tu-archivo",
                    #             info="Inserte el nombre del archivo de video, que está en la carpeta data/outputs",
                    #         )
                    with gr.Accordion(label="Crear representaciones", open=False):
                        # with gr.Row():
                        #     with gr.Column():
                        #         video_path_dataset = gr.Textbox(
                        #             label="Nombre del archivo de video.",
                        #             placeholder="Nombre-de-tu-archivo",
                        #             info="Inserte el nombre del archivo de video, que está en la carpeta data/outputs",
                        #         )
                                # annotation_file = gr.Textbox(label="Nombre del archivo de anotaciones.",
                                #     visible=True)
                                # path_embedds = gr.Textbox(
                                #     label="Nombre de la carpeta para guardar los personajes.",
                                #     placeholder="Nombre-de-tu-carpeta",
                                #     info="Inserte el nombre de la carpeta en la que guardar las representatciones de los personajes, luego será usada para predicir",
                                #     )
                                
                        # with gr.Row():
                        #     with gr.Accordion("Opciones avanzadas", open=False):
                        #         with gr.Column():
                        #             model = gr.Dropdown(
                        #                 choices=Finder.model_opts,
                        #                 label="Tipo de modelo para crear representaciones del personaje.",
                        #                 value=Finder.model_opts[0]
                        #             )
                        #         with gr.Column():
                        #             device = gr.Checkbox(
                        #                 label="cuda", info="En el caso de usar la gpu.(Recomendado)", value=True
                        #             )
                                    
                        with gr.Row():
                            embedds_button = gr.Button("Crear representaciones de los personajes")
                            # result_embedds = gr.Textbox(label="Resultado")
                

        # For the predict given the subs
        with gr.Tab("Predecir personajes"):

            with gr.Column(visible=True) as predict:
                # with gr.Row():
                #     with gr.Column():
                #         video_path = gr.Textbox(
                #             label="Nombre del archivo de video.",
                #             placeholder="Nombre-de-tu-archivo",
                #             info="Inserte el nombre del archivo de video, que está en la carpeta data/outputs",
                #         )

                # with gr.Row():
                #     with gr.Column():
                #         video_path = gr.Textbox(
                #             label="Nombre del archivo de video.",
                #             placeholder="Nombre-de-tu-archivo",
                #             info="Inserte el nombre del archivo de video, que está en la carpeta data/outputs",
                #         )
                        
                #         path_embedds_preds = gr.Textbox(
                #             label="Nombre de la carpeta donde están las representaciones de los personajes.",
                #             placeholder="Nombre-de-tu-carpeta",
                #             info="Inserte el nombre de la carpeta, que está en la carpeta data/outputs",
                #             )
                        
                            
            # with gr.Row():
                predict_button = gr.Button("Predecir los personajes")
                # predict_result = gr.Textbox(label="Resultado")
        
        # For creating dialogs and audio datasets
        with gr.Tab("Exportar para entrenamiento"):
            # select the function we want to use
            dataset_type = gr.Dropdown(
                choices=DatasetManager.dataset_types,
                label="Tipo de archivo para procesar",
                value=dataset_manager.dataset_type
            )
            # For the dialogs
            with gr.Column(visible=True) as dialogs:
                with gr.Row():
                    with gr.Column():
                        first_character = gr.Textbox(
                                label="Personaje uno, rol usuario.",
                                placeholder="Nombre-de-tu-archivo",
                                info="Nombre del personaje que hace las preguntas, en el prompt el rol de usuario",
                            )
                    with gr.Column():
                        second_character = gr.Textbox(
                                label="Personaje dos, rol sistema.",
                                placeholder="Nombre-de-tu-archivo",
                                info="Nombre del personaje que responde, en el prompt el rol de sistema.",
                            )
            # For the audios
            with gr.Column(visible=False) as audios:
                with gr.Row():
                    # with gr.Column():
                    #     audios_path = gr.Textbox(
                    #             label="Nombre de la carpeta de los audios.",
                    #             placeholder="Nombre-de-tu-archivo",
                    #             info="En la carpeta de outputs, al hacer las predicciones la carpeta donde se guardan las representaciones y los audios.",
                    #         )
                    with gr.Column():
                        character = gr.Textbox(
                                label="Personaje del que tomar los audios",
                                placeholder="Nombre-de-tu-archivo",
                                info="Nombre del personaje que responde, en el prompt el rol de sistema.",
                            )
            
            with gr.Row():
                with gr.Accordion("Opciones avanzadas", open=False):
                    with gr.Column():
                        time_interval = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Intervalo máximo entre diálogos, segundos",
                            interactive=True
                        )
                    # with gr.Column():
                    #     num_characters = gr.Slider(
                    #         minimum=1,
                    #         maximum=10,
                    #         value=4,
                    #         step=1,
                    #         label="Mínimo número de caracteres para usar la frase.",
                    #         interactive=True
                    #     )
            with gr.Row():
                transcribe_button = gr.Button("Transformar")
        
        # subtitles_type = gr.Textbox(visible=False, label="subtitles")
        
        
        # transcribe_button.click(
        #     call_function,
        #     # inputs=[
        #     #     dataset_type
        #     # ],
        #     outputs=[result_subs, annotation_file],
        # )

        # this is for the dialgs and audios
        transcribe_button.click(
            call_function,
            inputs=[
                dataset_type, transcribe
            ],
            outputs=[result],
        )
        
        # we will update the values of the dataset_manager class when the elements change
        # result_subs.change(
        #     update_visibility, inputs=[result_subs], outputs=[crop_labeling]
        # ).then()
        dataset_type.change(
            dataset_manager.update_dataset_type,
            inputs=[dataset_type],   
        ).then(update_visibility, inputs=[dataset_type], outputs=[dialogs, audios])

        
        subtitles_file.change(
            dataset_manager.update_subtitles_file,
            inputs=[subtitles_file],   
        )
        # is_crop.change(
        #     dataset_manager.update_crop,
        #     inputs=[is_crop],   
        # )
        
        # this is not updating well so there are errors, it may updates only in the first one, not in the last one
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
        # video_path_annotations.change(
        #     finder.update_video_path,
        #     inputs=[video_path_annotations],   
        # )
        # video_path_dataset.change(
        #     finder.update_video_path,
        #     inputs=[video_path_dataset],   
        # )
        # model.change(
        #     finder.update_model,
        #     inputs=[model],   
        # )
        # device.change(
        #     finder.update_device,
        #     inputs=[device],   
        # )
        
        # crop_label_button.click(
        #     finder.crop_for_labeling,
        #     # inputs=[path_labeling], 
        #     outputs=[result_crop_label]
        # )
        # load_data.click(
        #     load_df,
        #     inputs=[annotation_file], outputs=[dataframe]
        # )
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
            inputs=[model, device],outputs=[result])
        
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
    # config = load_global_config()
    # print(config)
    # dataset_manager = DatasetManager(
    #     dataset_type=config.dataset_manager.dataset_type,
    #     output_path=config.dataset_manager.output_path,
    #     num_characters=config.dataset_manager.num_characters,
    #     time_interval=config.dataset_manager.time_interval,
    #     audios_path=config.dataset_manager.audios_path,
    # )
    # print(dataset_manager)
    webui_dataset = create_ui()
    webui_dataset.launch()