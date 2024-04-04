import os
import pytest
from characterdataset.datasetmanager import DatasetManager


@pytest.fixture
def dataset_manager_instance():
    # Initialize DatasetManager with dummy arguments for testing
    dataset_manager = DatasetManager(
        dataset_type="dialogues",
        input_path="data/inputs/",
        subtitles_file="data/inputs/subtitles/Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC).srt",
        annotation_file="tests/data/outputs/[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 04 [WebRip 1080p HEVC-10bit AAC ASSx2]/[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 04 [WebRip 1080p HEVC-10bit AAC ASSx2].csv",
        output_path="tests/data/outputs",
        num_characters=4,
        time_interval=5,
        first_character="",
        second_character="クレア",
        character="クレア"
    )
    return dataset_manager

def test_inputs_check(dataset_manager_instance):
    # Test the inputs_check method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    
    # Call the method to be tested
    result = dataset_manager_instance.inputs_check()
    
    # Assert the expected outcome
    assert result == "Success"
    
    # Optionally, perform additional assertions to ensure correctness

def test_create_csv(dataset_manager_instance):
    # Test the create_csv method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    
    # Call the method to be tested
    result = dataset_manager_instance.create_csv(
        # subtitles_file="dummy_subtitles_file",
        # output_path="dummy_output_path",
        crop=True,
        # num_characters=4
    )
    
    # Assert the expected outcome
    assert result[0] == "Success"
    
    # Optionally, perform additional assertions to ensure correctness

def test_create_dialogues(dataset_manager_instance):
    # Test the create_dialogues method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    
    # Call the method to be tested
    result = dataset_manager_instance.create_dialogues(
        # annotation_file="dummy_annotation_file",
        # output_path="dummy_output_path",
        # time_interval=5,
        # num_characters=4,
        # first_character="dummy_first_character",
        # second_character="dummy_second_character"
    )
    
    # Assert the expected outcome
    assert result == "Dialogues have been created!"
    
    # Optionally, perform additional assertions to ensure correctness

def test_create_audio_files(dataset_manager_instance):
    # Test the create_audio_files method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    
    # Call the method to be tested
    result = dataset_manager_instance.create_audio_files(
        # annotation_file="dummy_annotation_file",
        # output_path="dummy_output_path",
        # num_characters=4,
        # character="dummy_character"
    )
    
    # Assert the expected outcome
    assert result == "Created audios of クレア"
    
    # Optionally, perform additional assertions to ensure correctness

@pytest.mark.skip(reason="テストを書いてる途中")
def test_transcribe_video(dataset_manager_instance):
    # Test the transcribe_video method
    
    # Setup any necessary preconditions
    # (e.g., create dummy files or directories)
    
    # Call the method to be tested
    result = dataset_manager_instance.transcribe_video(
        video_path="data/inputs/[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 01 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv",
        # output_path="dummy_output_path",
        iscropping=False,
        # num_characters=4
    )
    
    # Assert the expected outcome
    assert result[0] == "Transcrito audios!"
    
    # Optionally, perform additional assertions to ensure correctness
