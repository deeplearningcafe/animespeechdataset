import argparse
import os
import gradio as gr
import subprocess
import sys


import logging
from common.log import log
from configs import load_global_config

from characterdataset.datasetmanager import DatasetManager
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

def str_2_csv(dataset_manager: DatasetManager,
    intput_path:str=None, 
    output_path:str=None
    ) -> str:
    """ This function is used  to convert from str file to csv.

    Args:
        intput_path (str, optional): path of the str file. Defaults to None.
        output_path (str, optional): path to output the csv file. Defaults to None.

    Returns:
        str: success or failed message
    """
    # log.info("Start slicing...")
    
    # cmd = [
    #     "dataset_manager.py",
    #     "--dataset_type subtitles",
    #     "--subtitles_file",
    #     str(intput_path),
    #     "--output_path",
    #     str(output_path)
    # ]
    # log.info(f"Running: {' '.join(cmd)}")
    # result = subprocess.run(
    #     [python] + cmd,
    #     stdout=SAFE_STDOUT,  # type: ignore
    #     stderr=subprocess.PIPE,
    #     text=True,
    #     encoding="utf-8",
    # )
    # if result.returncode != 0:
    #     log.error(f"Error: {' '.join(cmd)}\n{result.stderr}")
    #     return "Error"
    # elif result.stderr:
    #     log.warning(f"Warning: {' '.join(cmd)}\n{result.stderr}")
    # log.success(f"Success: {' '.join(cmd)}")

    # return "Sucess"
    
    result = dataset_manager.create_csv(intput_path, output_path)
    
    return result


def call_function(dataset_type:str) -> str:
    log.info(f"Calling function of {dataset_type}")
    if dataset_type == DatasetManager.dataset_types[0]:
        result = dataset_manager.create_csv()
    elif dataset_type == DatasetManager.dataset_types[1]:
        result = dataset_manager.create_dialogues()
    elif dataset_type == DatasetManager.dataset_types[2]:
        result = dataset_manager.create_audio_files()

    else:
        return "Función desconocida"

    return result  
    

def update_visibility(dataset_type:str) -> None:
    if dataset_type == DatasetManager.dataset_types[1]:
        return gr.Column(visible=True), gr.Column(visible=False)
    elif dataset_type == DatasetManager.dataset_types[2]:
        return gr.Column(visible=False), gr.Column(visible=True)

def create_ui():

    with gr.Blocks() as dataset_app:
        
        # select the function we want to use
        dataset_type = gr.Dropdown(
            choices=DatasetManager.dataset_types,
            label="Tipo de archivo para procesar",
            value=dataset_manager.dataset_type
        )
        
        # Base case
        with gr.Row() as base:
            with gr.Column():
                subtitles_file = gr.Textbox(
                        label="Nombre del archivo de subtítulos.",
                        placeholder="Nombre-de-tu-archivo",
                        info="Inserte el nombre del archivo str, que está en la carpeta data/inputs",
                    )
                # output_path = gr.Textbox(
                #         label="Nombre del archivo de subtítulos.",
                #         placeholder="Nombre-de-tu-archivo",
                #         info="Inserte el nombre del archivo str, que está en la carpeta data/inputs",
                #     )
                
            with gr.Column():
                annotation_file = gr.Textbox(
                        label="Nombre del archivo de anotaciones.",
                        placeholder="Nombre-de-tu-archivo",
                        info="Inserte el nombre del archivo csv, que está en la carpeta data/outputs",
                    )
        
        # For the dialogs
        with gr.Column(visible=False) as dialogs:
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
                    with gr.Column():
                        num_characters = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Mínimo número de caracteres para usar la frase.",
                            interactive=True
                        )
        # For the audios
        with gr.Column(visible=False) as audios:
            with gr.Row():
                with gr.Column():
                    audios_path = gr.Textbox(
                            label="Nombre de la carpeta de los audios.",
                            placeholder="Nombre-de-tu-archivo",
                            info="En la carpeta de outputs, al hacer las predicciones la carpeta donde se guardan las representaciones y los audios.",
                        )
                with gr.Column():
                    character = gr.Textbox(
                            label="Personaje del que tomar los audios",
                            placeholder="Nombre-de-tu-archivo",
                            info="Nombre del personaje que responde, en el prompt el rol de sistema.",
                        )
            with gr.Row():
                with gr.Accordion("Opciones avanzadas", open=False):
                    with gr.Column():
                        num_characters = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Mínimo número de caracteres para usar la frase.",
                            interactive=True
                        )
                
                
        with gr.Row():
            result = gr.Textbox(label="Resultado")
            transcribe_button = gr.Button("Transformar")
        
        
        transcribe_button.click(
            call_function,
            inputs=[
                dataset_type
            ],
            outputs=[result],
        )
        
        # we will update the values of the dataset_manager class when the elements change
        dataset_type.change(
            dataset_manager.update_dataset_type,
            inputs=[dataset_type],   
        ).then(update_visibility, inputs=[dataset_type], outputs=[dialogs, audios])
        
        subtitles_file.change(
            dataset_manager.update_subtitles_file,
            inputs=[subtitles_file],   
        )
        annotation_file.change(
            dataset_manager.update_annotation_file,
            inputs=[annotation_file],   
        )
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
        num_characters.change(
            dataset_manager.update_num_characters,
            inputs=[num_characters],   
        )
        audios_path.change(
            dataset_manager.update_audios_path,
            inputs=[audios_path],   
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