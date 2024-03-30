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
from characterdataset import configs
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import json
import argparse
import os
from characterdataset.common import log

# Get the absolute path of the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = "default_config.toml"
CONFIG_PATH = os.path.join(current_dir, CONFIG_FILE)
OUTPUT_PATH = "data/outputs/train_llm"
CACHE_DIR = "pretrained_models/"

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
        cache_dir="pretrained_models/"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="pretrained_models/")

    # check if the tokenizer has padding token
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    formatting_func = formatting_prompts_func
    return formatting_func



def train(config_path):
    config = configs.load_global_config(config_path)
    torch.cuda.empty_cache()

    # load model and tokenizer
    model, tokenizer = load_model(
        model_id=config.train.base_model, 
        )
    
    modules = find_all_linear_names(model)
    # LoRAのパラメータ
    lora_config = LoraConfig(
        r=config.peft.rank,
        lora_alpha=config.peft.alpha,
        lora_dropout=config.peft.dropout,
        target_modules=modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


    # load dataset
    dataset = load_dataset_from_csv(config.dataset.dataset)

    # トレーナーの準備
    training_args = TrainingArguments(
            max_steps=config.train.max_steps,
            learning_rate=config.train.learning_rate,
            logging_steps=config.train.logging_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=config.train.save_steps,
            output_dir=f"{OUTPUT_PATH}/{config.dataset.character_name}/lora_{config.train.base_model}",
            save_total_limit=config.train.save_total_limit,
            push_to_hub=False,
            warmup_ratio=config.train.warmup_ratio,
            lr_scheduler_type=config.train.lr_scheduler_type,
            gradient_checkpointing=config.train.gradient_checkpointing,
            per_device_train_batch_size=config.train.per_device_train_batch_size,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            max_grad_norm=config.train.max_grad_norm,
            optim=config.train.optimizer,
            logging_dir=f"{OUTPUT_PATH}/{config.dataset.character_name}/logs"
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

def main(args):
    
    # checking if config file exists
    if not os.path.isfile(args.config_file):
        raise ValueError(
                    f"The config file at {args.config_file} does not exists"
                )
    
    train(args.config_file)
    log.info(f"Train completed!")
