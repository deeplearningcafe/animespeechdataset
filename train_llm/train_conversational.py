import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,)
import bitsandbytes as bnb
import accelerate
import pandas as pd
from datasets import Dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
import gradio as gr
from characterdataset import configs
from characterdataset.configs.config import save_global_config
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import json

CONFIG_PATH = "train_llm\default_config.toml"

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_peft_model(model_id:str, local_files:bool=False,
                    rank:int=64, alpha:int=64, dropout:float=0.1) -> tuple[PeftModel, AutoTokenizer]:
    # quantification configuration, 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        # load in the first cuda device
        device_map={"":0},
        trust_remote_code=True,
        local_files_only=local_files,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load peft model
    modules = find_all_linear_names(model)
    # LoRAのパラメータ
    lora_config = LoraConfig(
        r= 64,
        lora_alpha=64,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # モデルの前処理
    model = prepare_model_for_kbit_training(model, True)

    # LoRAモデルの準備
    model = get_peft_model(model, lora_config)

    # 学習可能パラメータの確認
    model.print_trainable_parameters()

    return model, tokenizer


def load_model(model_id:str, local_files:bool=False,) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # quantification configuration, 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        # load in the first cuda device
        device_map={"":0},
        trust_remote_code=True,
        local_files_only=local_files,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def load_dataset_from_csv(df_path:str) -> Dataset:
    df = pd.read_csv(df_path, header=0)

    df = df.rename(columns={"User": "input", "Assistant": "output"})

    dataset = Dataset.from_pandas(df, )

    return dataset

def load_formatting_func(config):

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            # text = f"USER:クレアになりきってください\nASSISTANT:はい、分かりましたわ。<|endoftext|>\nUSER:{example['input'][i]}\nASSISTANT: {example['output'][i]}<|endoftext|>"
            text = f"USER:{config.dataset.character_name}になりきってください。\nUSER:{example['input'][i]}\nASSISTANT:{example['output'][i]}<|endoftext|>"

            output_texts.append(text)
        return output_texts
    
    formatting_func = formatting_prompts_func()
    return formatting_func



def train():
    config = configs.load_global_config(CONFIG_PATH)
    torch.cuda.empty_cache()

    # load model and tokenizer
    model, tokenizer = load_model(
        model_id=config.train.base_model, 
        )
    
    modules = find_all_linear_names(model)
    # LoRAのパラメータ
    lora_config = LoraConfig(
        rank=config.peft.rank,
        alpha=config.peft.alpha,
        dropout=config.peft.dropout,
        target_modules=modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


    # load dataset
    dataset = load_dataset_from_csv(config.train.dataset)


    # Define  training args
    save_steps = 10
    logging_steps = 2
    max_steps=60

    # トレーナーの準備
    training_args = TrainingArguments(
            max_steps=max_steps,
            learning_rate=2e-4,
            logging_steps=logging_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=save_steps,
            output_dir="lora_clair_base",
            save_total_limit=3,
            push_to_hub=False,
            warmup_ratio=0.05,
            lr_scheduler_type="constant",
            gradient_checkpointing=True,
            per_device_train_batch_size=6,
            gradient_accumulation_steps=4,
            max_grad_norm=0.3,
            optim='adamw_8bit',
            logging_dir="train_llm/logs"
    )

    response_template = "ASSISTANT:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    formatting_prompts_func = load_formatting_func(config)
    
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        data_collator=collator,
        peft_config=lora_config
    )
    trainer.train()

    # save logs
    with open("train_llm/logs/states.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f)
    
    return "Train completed"

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
    config.dataset.dataset_path = dataset_path
    config.dataset.character_name = character_name

    config.train.max_steps = max_steps
    config.train.learning_rate = learning_rate * 1e-3
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
                                value=2,
                                minimum=1,
                                maximum=64,
                                step=1,
                            )
                            max_steps = gr.Slider(
                                label="Max number of steps",
                                info="1 step is one batch, so if the batch is 8 then one step consists of 8 data samples",
                                value=60,
                                minimum=10,
                                maximum=200,
                                step=5,
                            )
                            save_steps = gr.Slider(
                                label="Save steps",
                                info="Steps interval to save the model",
                                value=10,
                                minimum=1,
                                maximum=30,
                                step=2,
                            )
                            learning_rate = gr.Slider(
                                label="Learning rate",
                                info="This value will multiply 1e-3, so for learning rate of 1e-4 then value=10",
                                value=10,
                                minimum=1,
                                maximum=50,
                                step=1,
                            )
                            logging_steps = gr.Slider(
                                label="Save steps",
                                info="Steps interval to save the model",
                                value=4,
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
                                value=3,
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
                                value=4,
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
                            label="Learning rate",
                            info="This value will multiply 1e-3, so for learning rate of 1e-4 then value=10",
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
            train = gr.Button("Train the LLM")
        
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

    train.click(
        train,
        outputs=[result]
    )



    return train_app

if __name__ == "__main__":

    webui_training = create_ui()
    webui_training.launch()