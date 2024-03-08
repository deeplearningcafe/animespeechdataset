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

