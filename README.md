[[Japanese](README_jp.md)/English]


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
7. [License](#license)

## Introduction
`AnimeSpeech` is a project designed to generate datasets for training language models (LLMs) and text-to-speech (TTS) synthesis from anime subtitles. This project is aimed at making it easy to create the necessary data for training machine learning models using subtitles from anime videos. Specifically, it provides functionalities such as extracting conversation data and synthesizing character voices, which are useful for research and development in language modeling and speech synthesis.

Speaker recognition from videos, known as speaker verification task, unfortunately has not been used in this case. The current approach involves extracting embeddings from videos, while the character creation part involves human labeling of subtitles. Based on these labeled embeddings, they serve as training data for KNN. When predicting from new videos, it measures the distance between the labeled embeddings and the new embeddings, recognizing them as characters if the distance is smaller than a certain threshold.

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
│   │   ├── common
│   │   ├── configs
│   │   ├── datasetmanager
│   │   ├── oshifinder
├── tests
│   ├── test_dataset_manager.py
│   ├── test_finder.py
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
cd animespeechdataset

```
In case of using `conda` it is recommended to create a new environment.
```bash
conda create -n animespeech python=3.11
conda activate animespeech
```

Then install the required packages. In the case you don't have a nvidia gpu in your pc, then remove the `--index-url` line from the requirements file. As that line installs cuda software.
```bash
pip install -r requirements.txt
pip install -e .
```

To use the webui just run:
```bash
python webui_finder.py
```
### Creating character representations
1. Introduce the video name and the subtitles name, both placed in `data/inputs`. In the case of not having the subtitles, then use the transcribe checkbox.
2. Create reprentations(embeddings) of the desired characters, to load the dataset just use the `load df` button. 
3. The user labels the dataframe, just introduce the character name in the first column.
4. Save the labeled data using the `safe annotations` button.
5. Use the `Create representation` button to extract the embeddings from the labeled data.

### Predict characters
1. Introduce the video name and the subtitles name, both placed in `data/inputs`. In the case of not having the subtitles, then use the transcribe checkbox.
2. Use the `Predict characters` button, the annotation file path will be displayed at the annotation file textbox. The result file will be stored in a folder with the same name as the video file.

### Create audio and dialogs datasets
1. Introduce the prediction results file, the folder in which is stored should be included, but not the `data/outputs` part.
2. Select in the `Export for training` tab the type of dataset to create, `dialogues` or `audios`.
3. In the case of `dialogues` you can specify `first character` and `second character`, user role and assystant role. In the case of `audios` you have to choose the character.
4. For the `dialogues` you can choose the maximum time interval to consider 2 lines as a conversation, default is 5 seconds.
5. Click the `Transform` button.

### Transcribe
For speech recognition, we are using the nemo model released by reazonspeech. However, this module cannot be used directly on Windows. There are no issues when using WSL2. Therefore, we have included a simple script asr_api.py using FastAPI for speech recognition.

### Check best K for KKN
To look for the best `n_neighbors`, just run:
```bash
python -m characterdataset.oshifinder.knn_choose
```

## TODO
- [ ] Change classes attributes to function parameters when possible.  
- [X] When creating dialogues, look for (可能) with the character name.  
- [ ] Add support for Whisper.  
- [ ] Process entire folders, not just individual files.  
- [ ] Add QLoRA script for finetunning LLM.  
- [ ] Add the `n_neighbors` as parameter.  




## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License

This project is licensed under the MIT license. Details are in the [LICENSE](LICENSE) file.

