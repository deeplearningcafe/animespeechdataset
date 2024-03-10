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

The common and config part should be included in the src probably, as there is an error when importing.