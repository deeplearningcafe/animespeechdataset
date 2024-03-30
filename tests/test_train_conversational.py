import os
import pytest
import characterdataset.train_llm.train_conversational
from characterdataset.configs import load_global_config
import argparse
import pandas as pd
from datasets import Dataset

# Define paths to mock data or adjust to your file structure
CONFIG_PATH = "path/to/your/default_config.toml"
CSV_PATH = "path/to/your/dataset.csv"

# Fixture to provide a valid config path for testing
@pytest.fixture
def valid_config_path():
    # config_path = os.path.join(tmpdir, "data/default_config.toml")
    config_path = "data/default_config.toml"

    # with open(config_path, "w") as f:
    #     f.write("[train]\nbase_model = 'your_model_name'")
    return config_path

@pytest.fixture
def valid_config(valid_config_path):
    config_file = load_global_config(valid_config_path)

    return config_file

# Fixture to provide a valid CSV dataset for testing
@pytest.fixture
def valid_csv_dataset(valid_config):
    
    dataset = characterdataset.train_llm.train_conversational.load_dataset_from_csv(valid_config.dataset.dataset)

    # with open(csv_path, "w") as f:
    #     f.write("User,Assistant\nInput1,Output1\nInput2,Output2")
    return dataset

@pytest.fixture
def valid_format_func(valid_config):    
    example = pd.DataFrame({"input": ["この力をいかに活用するかで各国は競い合っている"], "output": ["そして魔道具の要となるのが"]})

    dataset = Dataset.from_pandas(example, )

    
    formatting_func = characterdataset.train_llm.train_conversational.load_formatting_func(valid_config)
    output_texts = formatting_func(dataset)

    return output_texts, valid_config.dataset.character_name

# Test load_model function
def test_load_model():
    model, tokenizer = characterdataset.train_llm.train_conversational.load_model(model_id="lightblue/karasu-1.1B")
    assert model is not None
    assert tokenizer is not None

# Test load_dataset_from_csv function
def test_load_dataset_from_csv(valid_csv_dataset):
    assert len(valid_csv_dataset) == 31
    assert "input" in valid_csv_dataset.features
    assert "output" in valid_csv_dataset.features

# Test load_formatting_func function
def test_load_formatting_func(valid_format_func):
    text, name = valid_format_func
    assert type(text[0]) == str
    assert text[0] == f"USER:{name}になりきってください。\nUSER:この力をいかに活用するかで各国は競い合っている\nASSISTANT:そして魔道具の要となるのが<|endoftext|>"

# Test train function
@pytest.mark.skip(reason="テストを書いてる途中")
def test_train(valid_config_path):
    # Provide valid config path and CSV dataset
    args = argparse.Namespace(config_file=valid_config_path)
    train_output = characterdataset.train_llm.train_conversational.train(args)
    assert train_output == "Train completed"

# Test main function
@pytest.mark.skip(reason="テストを書いてる途中")
def test_main(valid_config_path, caplog):
    # Provide valid config path and use caplog to capture log messages
    args = argparse.Namespace(config_file=valid_config_path)
    characterdataset.train_llm.train_conversational.main(args)
    assert "Train completed!" in caplog.text
