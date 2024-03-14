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

このプロジェクトはMITライセンスの下でライセンスされています。詳細については[LICENSE.md](LICENSE)ファイルを参照してください。





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

Solve the error in cropping for labels. -> When there are several elements with the same name, only the last one counts for updating.
As there are 3 annotation files variables, only the last one is used to update.

Clean the innecesary stuf, repeated variables are problematic so only unique ones.
For the case of creating labeled data, the user inputs are: Subs file, video file (the output folder for labeling should be automatically created and then removed), name for the character embeddings folder probably should be automatic as well.

We could remove the subtitle functionality and make all the functions take as input the str, then internally convert it.
We should leave only 3 buttons, one for starting to label(this converts the sub file, creates the temp folder with the audios), the second button should be the save annotations(save the updates csv and also removes the temp folder), finally, the last button should be create the embedds.

The path of the audios for labeling should be simple to make it easier to copy paste them.

There is an error that we can't select models and device for the embedds, probably because it is not "interactive".

## Webui_dataset
We should unify all the webui in only one, but we can develop them alone and then just merge.
For creating dialogs the user needs to introduce the annotation file, character 1 (user role) and character 2 (system role).
For creating the audios the user needs to introduce the annotation file, character name and the audios folder.
The audios folder should be automatically created, we can include it in outputs with the name of the character. But there is a problem if after getting the audios we use other annotation file, then the previous audios will get removed. Because the index is the same, so we need to manage the index part. We can create a folder with the annotation file name inside the character folder. That shouldn't be a problem for the tts.
Creating audio folder for each annotation file is good, but we should append to just 1 text file. Other better option is to get the last index and just continue adding. But it is difficult because the index is used for looking for the audio file. 
We can use as name for the audios the index and the text.

The preds file is already in the same folder as the audios so we can get the audios_path from there. 
There is a problem, as the preds is in the folder of the video, we need to include that folder name as well.

We need to solve the problem of including the folder of the annotation file, as the folder name should be the same as the csv file, we can get it from there as well as we did with the audios paths.

# Webui_finder
## Subtitles
Given the str file with the subtitles, transform that file to csv file format.
## Crop for labeling
Given a csv file with the times and the text, creates clips audios from the video and a csv file to include the character of each line.
With the labeled csv we can create the embeddings for predicting.
## Create embeddings
From the labeled csv created in crop for labeling, we create the embeddings for each character.
It does not return any comment like "completed" or "error".
## Predict
In the actual implementation, we first need to convert the str file in the subtitles part, then we use the new csv file, the video and the
path of the embeddings to predict the characters of the csv file.
It would be better to use as input the str file for this part, to make it more simpler, so that it automatically converts to csv file in the process.

As there are several variables shared among the different methods, one option is to leave those vars at the start so that they are always visible and just with accordeons appear the specific variables.
Probably this is the best option I think.

The folder of character_embedds should be in data, but not in inputs nor in outputs. We could specify from the beggning the folder to save the embedds.

The create labeling data could be done as the predict button, first call the call_function and then use the updated annotation file.

Should the transcribe logic be all in the dataset_manager? In that class we have alredy stored the "iscropping" and "num_characters". Also the segments_2_annotations is in the text_dataset.py. We can move the utils.py in common to have there ffmpeg functions and that stuf.

We could processes the response of the api, in json format, in the text_dataset.py, so that we have the text processing in only one file.

We need to normalize the path of the audio when sending to the API

IMPORTANT: The common and config part should be included in the src probably, as there is an error when importing.



There is an error that when creating the dialogs, it only takes the character wroten, not the (可能) as well.

In character creation, the "cropping" option should always be true, as there we want to create the embeddings for predicting so the character column is always necessary.
We need to update the cropping logic.

The model selection in predictic works but in create embedds does not.
That is because the name of the variable is the same as in the predicts.
Merge all the repeated advanced options.

The filename column should be removed as the tmp audio files are deleted.

Maybe we should include a script to download the models automatically.

Include a script that creates the folder if they not exists?

The min characters is not working. It can't be updated.

There was a problem with the predict when saving the csv because of the csv filename, so make it just preds.csv.

What is the objective of min length text in the dialogs and audios? This functions use the predictions, which use the sub transformed. So the min characters should only be used for the subtitles transform.

We should change the log warnings to raise value error?

The implementation of update annotation_file is differenct for finder and for dataset_manager. But as the one in finder should be automatic, there is not problem with the actual implementation.

Change the log by Raising errors, because I don't know the line where the error was produced.

# Installation
We need to include the soundfile.