# AnimeSpeech: Dataset Generation for Language Model Training and Text-to-Speech Synthesis from Anime Subtitles

This project aims to facilitate the generation of datasets for training Language Models (LLMs) and Text-to-Speech (TTS) synthesis using subtitles from anime videos.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Inputs](#inputs)
4. [Functionalities](#functionalities)
    - [Create Annotations](#Create-Annotations)
    - [Create Datasets](#Create-Datasets)
5. [Directory Structure](#directory-structure)
6. [How to Use](#how-to-use)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

## Requirements
- demoji==1.1.0
- gradio==4.20.1
- matplotlib==3.8.0
- munch==4.0.0
- neologdn==0.5.2
- numpy==1.25.2
- pandas==2.0.3
- scikit_learn==1.2.2
- setuptools==69.1.1
- speechbrain==0.5.16
- toml==0.10.2
- torch==2.0.1
- torchaudio==2.0.2
- tqdm==4.66.1
- transformers==4.36.2


## Inputs
- **Video file**: The video from which audio and dialog data will be extracted.
- **Subtitles file**: The .str file containing the subtitles of the video.
- **Annotation file**: The csv file containing the predictions. This file should be used only for creating audio and dialogs datasets, for labeling or predicting it should be outputed automatically.

Both the video file and subtitles file should be placed in the `data/inputs` folder. Only the filenames are required; the full path is not needed. In the case of the annotation file, the path of the folder is needed. For example:
`video_name/preds.csv` which is in the `data/outputs` folder.  

## Functionalities
### Create Annotations
This functionality involves processing subtitles and video to generate annotations.

#### Character Creation
Users can label the data to create representations of desired characters. The converted subtitles are transformed into tabular data similar to an Excel sheet. For predicting new data, we need the embeddings of the characters so, this process needs to be done at least once.

#### Character Prediction
This function predicts the character speaking each line. It requires pre-created representations (embeddings) of desired characters and predicts characters only for those with representations.

### Create Datasets
This functionality takes an annotations file as input and creates datasets for training LLMs and TTS.

#### Dialogues Dataset
This creates a conversational dataset suitable for training LLMs. Users can select which characters' dialogues to include.

#### Audios Dataset
This extracts all audios of a desired character and organizes them into a folder along with corresponding text for TTS training.

## Directory Structure
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
│   │   ├── datasetmanager
│   │   ├── oshifinder
├── webui_finder.py
```
### File description
    -data stores the subtitles and video files, the predictions get saved there as well.
    -datasetmanager sub-package that processes subtitles files and the text part.
    -oshifinder sub-package that creates embeddings and makes predictions.
    -webui_finder.py gradio based interface.

## How to Use

### Installation
```bash
git clone https://github.com/deeplearningcafe/animespeechdataset
pip install -r requirements.txt
pip install -e .
```

To use the webui just run:
```python
python webui_finder.py
```
#### Creating character representations
1. Introduce the video name and the subtitles name, both placed in `data/inputs`. In the case of not having the subtitles, then use the transcribe checkbox.
2. Create reprentations(embeddings) of the desired characters, to load the dataset just use the `load df` button. 
3. The user labels the dataframe, just introduce the character name in the first column.
4. Use the `Create representation` button to extract the embeddings from the labeled data.

#### Predict characters
1. Introduce the video name and the subtitles name, both placed in `data/inputs`. In the case of not having the subtitles, then use the transcribe checkbox.
2. Use the `Predict characters` button, the annotation file path will be displayed at the annotation file textbox. The result file will be stored

#### Create audio and dialogs datasets
1. Introduce the prediction

## Contributing

[Explain how others can contribute to the project. This could include information on submitting bug reports, feature requests, or pull requests.]

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License

このプロジェクトはMITライセンスの下でライセンスされています。詳細については[LICENSE.txt](LICENSE)ファイルを参照してください。





# CSV from str
Use dataset.py 

python crop.py --annotate_map datasets\rei_clair.csv --role_audios test --video_path wataoshi_1.mkv --model wavlm --device cuda

python predict.py --annotate_map datasets\wataoshi3.csv --save_folder tmp --video_path "E:\Data\LLM\わたおし\[LoliHouse] Watashi no Oshi wa Akuyaku Reijou [01-12][WebRip 1080p HEVC-10bit AAC]\[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 03 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv" --output_path datasets\preds\preds_wataoshi3.csv --character_folder role_audios/feature

We want to create databases for first the LLM, that is instruction tunning so we want convertional format, that is question and answer.
Then we want a database for the TTS, that is audio files with their names and their texts.

python dataset_manager.py --dataset_type subtitles --subtitles_file "E:\Data\LLM\わたおし\私の推しは悪役令嬢。\Watashi no Oshi wa Akuyaku Reijou. - 03 「私の恋は七転八起。」 (AT-X 1280x720 x264 AAC).srt" --output_path test

python dataset_manager.py --dataset_type dialogues --annotation_file "datasets\preds\preds_wataoshi3.csv" --output_path test
python dataset_manager.py --dataset_type audios --annotation_file "datasets\preds\preds_wataoshi3.csv" --output_path test --audios_path "tmp\[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 03 [WebRip 1080p HEVC-10bit AAC ASSx2]\voice" --character "クレア"


data\outputs\Watashi no Oshi wa Akuyaku Reijou. - 04 「魔物の襲撃は油断大敵。」 (AT-X 1280x720 x264 AAC).csv
[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 04 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv
character_embedds\embeddings

For dialogs and audios:
[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 04 [WebRip 1080p HEVC-10bit AAC ASSx2]\[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 04 [WebRip 1080p HEVC-10bit AAC ASSx2]_preds.csv
レイ
クレア
[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 04 [WebRip 1080p HEVC-10bit AAC ASSx2]\voice

# Todo:
Make more clear the outputs and inputs folders.
Change the update in classes to parameters in functions when needed.
Improve the labeling part.


## Webui_dataset
We should unify all the webui in only one, but we can develop them alone and then just merge.
For creating dialogs the user needs to introduce the annotation file, character 1 (user role) and character 2 (system role).
For creating the audios the user needs to introduce the annotation file, character name and the audios folder.
The audios folder should be automatically created, we can include it in outputs with the name of the character. But there is a problem if after getting the audios we use other annotation file, then the previous audios will get removed. Because the index is the same, so we need to manage the index part. We can create a folder with the annotation file name inside the character folder. That shouldn't be a problem for the tts.
We can use as name for the audios the index and the text.

There is a problem, as the preds is in the folder of the video, we need to include that folder name as well.


# Webui_finder
## Subtitles
Given the str file with the subtitles, transform that file to csv file format.
## Crop for labeling
Given a csv file with the times and the text, creates clips audios from the video and a csv file to include the character of each line.
With the labeled csv we can create the embeddings for predicting.
## Create embeddings
From the labeled csv created in crop for labeling, we create the embeddings for each character.
## Predict
In the actual implementation, we first need to convert the str file in the subtitles part, then we use the new csv file, the video and the
path of the embeddings to predict the characters of the csv file.






There is an error that when creating the dialogs, it only takes the character wroten, not the (可能) as well.





Maybe we should include a script to download the models automatically.

Include a script that creates the folder if they not exists?


What is the objective of min length text in the dialogs and audios? This functions use the predictions, which use the sub transformed. So the min characters should only be used for the subtitles transform.


The implementation of update annotation_file is differenct for finder and for dataset_manager. But as the one in finder should be automatic, there is not problem with the actual implementation.
There is an error with the annotation file for creating dialogs, as the implementation is different, we can't use directly the result of the predction, we need to remove the path of the generated annotation file. The problem is that the annotation file is not the same as the prediction file.
But everything else works.

Change the code logic, classes have many attributes, we should use parameters in the functions. It makes no sense to be storing attributes and then use them as paramters to other functions.

Update clases attributes to getters and setters.

In gradio the then() always continues despite the result of the first part.

# Installation
We need to include the soundfile.