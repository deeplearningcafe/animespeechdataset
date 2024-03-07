from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio.transforms as T
import torchaudio


# audio_embds_dir = "role_audios/feature"
audio_embds_dir = "test\embeddings"

embeddings = None
labels = []
dim = 0
# これはサブフォルダの名前をリストに格納する
role_dirs = []
for item in os.listdir(audio_embds_dir):
    if os.path.isdir(os.path.join(audio_embds_dir, item)):
        role_dirs.append(item)

for role_dir in role_dirs:
    print(f'{audio_embds_dir}/{role_dir}')

    role = os.path.basename(os.path.normpath(role_dir))
    
    file_list = [os.path.join(audio_embds_dir, role_dir, embeddings_path) for embeddings_path in os.listdir(os.path.join(audio_embds_dir, role_dir))]
    
    for embeddings_path in file_list:
        with open(embeddings_path, 'rb') as fp:
            embedding = pickle.load(fp)
        fp.close()
        
        # 前作ったリストに格納する
        if dim == 0:
            embeddings_cls = embedding
            dim = embeddings_cls.shape[0]
        else:
            # This is equivalent to concatenation along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N)
            embeddings_cls = np.vstack((embeddings_cls, embedding))
        
        labels.append(role)

print(embeddings_cls.shape, len(labels))


#t-SNEで次元削減
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
if len(embeddings_cls.shape) > 2:
    embeddings_cls = embeddings_cls.squeeze(1)
    
normed_embedding = tsne.fit_transform(embeddings_cls)
print(normed_embedding.shape)


fig = plt.figure(figsize=(10, 10))

colors = {f"{next(iter(set(labels)))}": "red", f"{list(set(labels))[-1]}": "blue"}
print(colors)
for i in range(normed_embedding.shape[0]):
    plt.scatter(normed_embedding[i, 0], normed_embedding[i, 1],
                c=colors[labels[i]])
    
plt.xlim((normed_embedding[:, 0].min(), normed_embedding[:, 1].max()))
plt.ylim((normed_embedding[:, 1].min(), normed_embedding[:, 1].max()))

plt.show()


exit()


def srt_format_timestamp( seconds):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours:02d}:") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"



# csvから前処理を行う
input_video = r"E:\Data\LLM\わたおし\[LoliHouse] Watashi no Oshi wa Akuyaku Reijou [01-12][WebRip 1080p HEVC-10bit AAC]\[LoliHouse] Watashi no Oshi wa Akuyaku Reijou - 03 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv"
input_str = "datasets\wataoshi3.csv"
temp_folder = "test"
df = pd.read_csv(input_str, header=0)
print(df)

voice_dir = 'voice'
file = os.path.basename(input_video)
filename, format = os.path.splitext(file)
os.makedirs(f'{temp_folder}/{filename}/{voice_dir}', exist_ok=True)
print(f'{temp_folder}/{filename}/{voice_dir}')


# 録音データを作る
for index in tqdm(range(len(df))):
    start_time, end_time, text = df.iloc[index, :]
    index = str(index).zfill(4)
    start_time = float(start_time)
    end_time = float(end_time)
    
    # ss = start_time.zfill(11).ljust(12, '0')[:12]
    # ee = end_time.zfill(11).ljust(12, '0')[:12]
    ss = srt_format_timestamp(start_time)
    ee = srt_format_timestamp(end_time)
    
    
    name = f'{index}_{ss}_{ee}_{text}'.replace(':', '.')

    audio_output = f'{temp_folder}/{filename}/{voice_dir}/{name}.wav'
    command = ['ffmpeg', '-ss',str(start_time), '-to', str(end_time), '-i', f'{input_video}', "-vn",
            '-c:a', 'pcm_s16le','-y', audio_output, '-loglevel', 'quiet']

    subprocess.run(command)


# 録音データから埋め込みを作る
file = os.path.basename(input_video)
filename, format = os.path.splitext(file)

temp_dir = f'{temp_folder}/{filename}'

# これはリストを返り値
voice_files = os.listdir(os.path.join(temp_dir, voice_dir))
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            run_opts={"device": "cuda"})


#埋め込みを作成する
for pth in tqdm(voice_files, f'extract {filename} audio features ,convert .wav to .pkl'):
    new_dir = os.path.join(temp_dir, 'feature')
    os.makedirs(new_dir, exist_ok=True)
    file = os.path.basename(pth)
    file, format = os.path.splitext(file)
    pth = os.path.join(temp_dir, voice_dir, pth)
    # print(pth)
    try:
        # サンプリングレートは16khzであるべき
        signal, fs = torchaudio.load(pth)
        # 録音の前処理
        signal_mono = torch.mean(signal, dim=0)
        # change freq
        resample_rate = 16000
        resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
        resampled_waveform = resampler(signal_mono)

        
        embeddings = classifier.encode_batch(resampled_waveform)

        # 埋め込みを保存する
        with open(f"{new_dir}/{file}.pkl", "wb") as f:
            pickle.dump(embeddings.detach().cpu(), f)
    except:
        continue

print("録音データから埋め込みを作成しました。")



knn_classifier = KNeighborsClassifier(n_neighbors=4, metric='cosine')
knn_classifier.fit(embeddings_cls.squeeze(1), labels)

threshold_certain = 0.4
threshold_doubt = 0.6

temp_embeds = os.path.join(temp_dir, 'feature')
file_list = [os.path.join(temp_embeds, embeddings_path) for embeddings_path in os.listdir(os.path.join(temp_embeds))]

preds = []
distances = []
for path in file_list:
    with open(path, 'rb') as fp:
        embedding = pickle.load(fp)
    fp.close()
    # print(embedding.shape)
    embedding = embedding.squeeze(0)
    
    predicted_label = knn_classifier.predict(embedding)
    dist, _ = knn_classifier.kneighbors(embedding)
    dist = dist[0].min()
    
    name = ''
    if dist < threshold_certain:
        name = predicted_label[0]
    elif dist < threshold_doubt:
        name = '(可能)' + predicted_label[0]
    
    preds.append(name)
    distances.append(dist)


df = pd.DataFrame({"filename": file_list, "predicted_label": preds, "distance": distances})
print(df.head())
df.to_csv("preds_wataoshi3.csv", index=False)