import gradio as gr
from characterdataset import configs
from characterdataset.configs.config import save_global_config
import json

from characterdataset.train_llm import train
CONFIG_PATH = r"src\characterdataset\train_llm\default_config.toml"


def update_config(
        base_model:str,
        dataset_path:str,
        character_name:str,
        max_steps:int,
        learning_rate:int,
        per_device_train_batch_size:int,
        optimizer:str,
        save_steps:int,
        logging_steps:int,
        save_total_limit:int,
        warmup_ratio:float,
        lr_scheduler_type:str,
        gradient_checkpointing:bool,
        gradient_accumulation_steps:int,
        max_grad_norm:float,
        rank:int,
        alpha:int,
        dropout:float,
    ):
    config = configs.load_global_config(CONFIG_PATH)

    config.train.base_model = base_model
    config.dataset.dataset = dataset_path
    config.dataset.character_name = character_name

    learning_rate = learning_rate * 1e-5
    config.train.max_steps = max_steps
    config.train.learning_rate = learning_rate
    config.train.per_device_train_batch_size = per_device_train_batch_size
    config.train.optimizer = optimizer
    config.train.save_steps = save_steps
    config.train.logging_steps = logging_steps
    config.train.save_total_limit = save_total_limit
    config.train.warmup_ratio = warmup_ratio
    config.train.lr_scheduler_type = lr_scheduler_type
    config.train.gradient_checkpointing = gradient_checkpointing
    config.train.gradient_accumulation_steps = gradient_accumulation_steps
    config.train.max_grad_norm = max_grad_norm
    config.peft.rank = rank
    config.peft.alpha = alpha
    config.peft.dropout = dropout

    # save the new configuration
    save_global_config(config, filepath=CONFIG_PATH)

    return "Configuration updated"
    


def create_ui():
    with gr.Blocks() as train_app:
        
        with gr.Row():
            with gr.Column():
                with gr.Tab("Training configuration"):
                    with gr.Row():
                        with gr.Column():
                            batch_size = gr.Slider(
                                label="Batch size",
                                info="学習速度が遅い場合は小さくして試し、VRAMに余裕があれば大きくしてください。JP-Extra版でのVRAM使用量目安: 1: 6GB, 2: 8GB, 3: 10GB, 4: 12GB",
                                value=16,
                                minimum=1,
                                maximum=64,
                                step=1,
                            )
                            max_steps = gr.Slider(
                                label="Max number of steps",
                                info="1 step is one batch, so if the batch is 8 then one step consists of 8 data samples",
                                value=80,
                                minimum=10,
                                maximum=200,
                                step=5,
                            )
                            save_steps = gr.Slider(
                                label="Save steps",
                                info="Steps interval to save the model",
                                value=5,
                                minimum=1,
                                maximum=30,
                                step=2,
                            )
                            learning_rate = gr.Slider(
                                label="Learning rate",
                                info="This value will multiply 1e-5, so for learning rate of 1e-4 then value=10",
                                value=10,
                                minimum=1,
                                maximum=50,
                                step=1,
                            )
                            logging_steps = gr.Slider(
                                label="Logging steps",
                                info="Steps interval to log the loss of the model",
                                value=5,
                                minimum=1,
                                maximum=30,
                                step=2,
                            )
                    with gr.Row():
                        with gr.Accordion("Advanced options", open=False):
                            optimizer = gr.Dropdown(
                                choices=["adamw_torch", "adafactor", "adamw_bnb_8bit"],
                                label="Type of optimizer, adamw_8bit reduces memory usage",
                                value="adamw_bnb_8bit"
                            )
                            save_total_limit = gr.Slider(
                                label="Number of models  to save",
                                info="Maximum number of trained models to store",
                                value=10,
                                minimum=1,
                                maximum=30,
                                step=1,
                            )
                            warmup_ratio = gr.Slider(
                                label="Number of models  to save",
                                info="Maximum number of trained models to store",
                                value=0.05,
                                minimum=0.0,
                                maximum=0.2,
                                step=0.01,
                            )
                            gradient_accumulation_steps = gr.Slider(
                                label="Number of models  to save",
                                info="Maximum number of trained models to store",
                                value=2,
                                minimum=1,
                                maximum=16,
                                step=1,
                            )
                            gradient_checkpointing = gr.Checkbox(
                                label="Number of models  to save",
                                info="Maximum number of trained models to store",
                                value=True,
                            )
                            max_grad_norm = gr.Slider(
                                label="Number of models  to save",
                                info="Maximum number of trained models to store",
                                value=0.3,
                                minimum=0.0,
                                maximum=2.0,
                                step=0.1,
                            )
                            lr_scheduler_type = gr.Dropdown(
                                choices=["linear", "cosine", "constant"],
                                label="Type of lr scheduler",
                                value="constant"
                            )

                    
                with gr.Tab("Peft configuration"):
                    with gr.Column():
                        rank = gr.Slider(
                            label="LoRA Rank",
                            info="Rank of the LoRA matrix",
                            value=64,
                            minimum=8,
                            maximum=128,
                            step=2,
                        )
                        alpha = gr.Slider(
                            label="LoRA Alpha",
                            info="Alpha for the LoRA matrix",
                            value=64,
                            minimum=8,
                            maximum=128,
                            step=2,
                        )   
                        dropout = gr.Slider(
                            label="Dropout",
                            info="This is the dropout probability for the LoRA modules",
                            value=0.1,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                        )
            with gr.Column():
                base_model = gr.Textbox(label="Name of the model to use")
                dataset_path = gr.Textbox(label="Name of the dialogues file")
                character_name = gr.Textbox(label="Name of the character")
                update = gr.Button("Update the configuration file")

        with gr.Row():
            result = gr.Textbox()
            train_button = gr.Button("Train the LLM")
        
        config_textbox = gr.Textbox(CONFIG_PATH, visible=False)
        
        update.click(
            update_config,
            inputs=[
                base_model,
                dataset_path,
                character_name,
                max_steps,
                learning_rate,
                batch_size,
                optimizer,
                save_steps,
                logging_steps,
                save_total_limit,
                warmup_ratio,
                lr_scheduler_type,
                gradient_checkpointing,
                gradient_accumulation_steps,
                max_grad_norm,
                rank,
                alpha,
                dropout,
            ],
            outputs=[result]
        )

        train_button.click(
            train,
            inputs=[config_textbox],
            outputs=[result]
        )



    return train_app

if __name__ == "__main__":

    webui_training = create_ui()
    webui_training.launch()