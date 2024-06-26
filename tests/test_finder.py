import os
import pytest
from characterdataset.oshifinder import Finder
from characterdataset.configs import load_global_config
import pandas as pd

config = load_global_config()

@pytest.fixture
def finder_instance():
    # Initialize Finder with dummy arguments for testing
    finder = Finder(
        input_path="data/inputs/",
        annotation_file="tests/data/outputs/Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC).csv",
        output_path="tests/data/outputs",
        video_path="data/inputs/videos/[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 04 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv",
        output_path_labeling="tests/data/outputs/tmp",
        character_folder="tests/data/character_embedds"
    )
    return finder

def test_crop_for_labeling(finder_instance):
    # Test the crop_for_labeling method
    
    # Setup any necessary preconditions
    finder_instance.annotation_file = "tests/data/outputs/Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC)_updated_updated.csv"
    df = pd.read_csv(finder_instance.annotation_file, header=0)
    print(df.columns)
    # Call the method to be tested
    result = finder_instance.crop_for_labeling()
    
    # Assert the expected outcome
    assert result == "Success"
    
    # Optionally, perform additional assertions to ensure correctness
    # check temp folder was created
    assert os.path.isdir(finder_instance.output_path_labeling)
    # check there are audios inside
    assert len(os.listdir(finder_instance.output_path_labeling)) > 0

def test_crop_files(finder_instance):
    # Test the crop_files method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    # finder_instance = finder_instance()
    finder_instance.annotation_file = r"tests/data/outputs/Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC)_updated.csv"

    
    # Call the method to be tested
    result = finder_instance.crop_files(model="speechbrain", device=True)
    
    # Assert the expected outcome
    assert result == "Characters embeddings have been created!"
    
    # Optionally, perform additional assertions to ensure correctness
    assert os.path.isdir(finder_instance.character_folder)
    # check there are character folders inside
    assert len(os.listdir(finder_instance.character_folder)) > 0

@pytest.mark.asyncio
async def test_make_predictions(finder_instance):
    # Test the make_predictions method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    # finder_instance = finder_instance()
    finder_instance.annotation_file = r"data/outputs/Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC).csv"
    
    # Call the method to be tested
    result = await finder_instance.make_predictions(model="speechbrain", device=True)
    
    # Assert the expected outcome
    assert result == "Predictions have been completed!"
    
    # Optionally, perform additional assertions to ensure correctness

# @pytest.mark.skip(reason="テストを書いてる途中")
def test_crop_files_api(finder_instance):
    # Test the crop_files method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    # finder_instance = finder_instance()
    finder_instance.annotation_file = r"tests/data/outputs/Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC)_updated_updated.csv"

    
    # Call the method to be tested
    result = finder_instance.crop_files(model="espnet", device=True)
    
    # Assert the expected outcome
    assert result == "Characters embeddings have been created!"
    
    # Optionally, perform additional assertions to ensure correctness
    assert os.path.isdir(finder_instance.character_folder)
    # check there are character folders inside
    assert len(os.listdir(finder_instance.character_folder)) > 0
    
# @pytest.mark.skip(reason="テストを書いてる途中")
@pytest.mark.asyncio
async def test_make_predictions_api(finder_instance):
    # Test the make_predictions method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    # finder_instance = finder_instance()
    finder_instance.annotation_file = r"data/outputs/Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC).csv"
    
    # Call the method to be tested
    result = await finder_instance.make_predictions(model="espnet", device=True)
    
    # Assert the expected outcome
    assert result == "Predictions have been completed!"
    
    # Optionally, perform additional assertions to ensure correctness