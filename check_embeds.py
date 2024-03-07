import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pickle
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from speechbrain.pretrained import SpeakerRecognition
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import os

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            run_opts={"device": "cuda"},)

audio_path = r"role_audios\voice\クレア\0024_00.03.12.550_00.03.17.060_んっ…。バカにしていますのレイ=テイラー!.wav"
signal, fs =torchaudio.load(audio_path)

# convert from stereo to mono
signal_mono = torch.mean(signal, dim=0)
print(signal_mono.shape)

# change freq
resample_rate = 16000
resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
resampled_waveform = resampler(signal_mono)
print(resampled_waveform.shape)

embeddings = classifier.encode_batch(resampled_waveform)
print("Freq: ", fs)
# audio_path = r"role_audios\feature\クレア\0050_00.04.43.240_00.04.46.950_ぼっとしていらっしゃるから置物かと思いましたわ。.wav.pkl"
audio_path = r"role_audios\feature\レイ\0031_00.03.35.740_00.03.38.750_私はクレア様が大好きです!.wav.pkl"

with open(audio_path, "rb") as fp:
    data = pickle.load(fp)
fp.close()
data = data.to("cuda")
data = torch.mean(data, dim=0).unsqueeze(0)
print(embeddings.shape, data.shape)

# score1 = torch.mean(torch.matmul(embeddings[:][0], data[:][0].transpose(-1, -2))) # 1
# score2 = F.cosine_similarity(embeddings[:][0], data[:][0])
# print(score1, score2)
# score1 = torch.mean(torch.matmul(embeddings.squeeze(1), data.squeeze(1).transpose(-1, -2))) # 1
# score2 = F.cosine_similarity(embeddings.squeeze(1), data.squeeze(1))
# print(score1, score2)

# data to mono

embeddings_norm = F.normalize(embeddings.squeeze(1), p=2, dim=1).detach().cpu()
data_norm = F.normalize(data.squeeze(1), p=2, dim=1).detach().cpu()
score1 = torch.mean(torch.matmul(embeddings_norm, data_norm.transpose(-1, -2))) # 1
score2 = F.cosine_similarity(embeddings_norm, data_norm)
print(score1, score2)
# print(embeddings[:][0].shape) torch.Size([1, 192])
# print(embeddings.squeeze(1).shape) torch.Size([2, 192])


verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               run_opts={"device": "cuda"})

# audio_path1 = r"role_audios\voice\クレア\0024_00.03.12.550_00.03.17.060_んっ…。バカにしていますのレイ=テイラー!.wav"
# audio_path2 = r"role_audios\voice\クレア\0050_00.04.43.240_00.04.46.950_ぼっとしていらっしゃるから置物かと思いましたわ。.wav"

audio_path1 = r"0024_00.03.12.550_00.03.17.060_んっ…。バカにしていますのレイ=テイラー!.wav"
audio_path2 = r"0050_00.04.43.240_00.04.46.950_ぼっとしていらっしゃるから置物かと思いましたわ。.wav"

score, prediction = verification.verify_files(audio_path1, audio_path2) # Different Speakers
print(score, prediction) # tensor([0.6216], device='cuda:0') tensor([True], device='cuda:0')
score, prediction = verification.verify_files(audio_path1, audio_path1) # Same Speaker
print(score, prediction) # tensor([1.0000], device='cuda:0') tensor([True], device='cuda:0')

audio_path3 = r"0023_00.03.09.380_00.03.12.550_私の名前覚えていらっしゃいますか.wav"

score, prediction = verification.verify_files(audio_path1, audio_path3) # Different Speakers
print(score, prediction) # tensor([0.3604], device='cuda:0') tensor([True], device='cuda:0')

print("*"*10)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus')
# model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus')
audio_path_1 = r"role_audios\voice\クレア\0024_00.03.12.550_00.03.17.060_んっ…。バカにしていますのレイ=テイラー!.wav"

signal_1, fs =torchaudio.load(audio_path_1)

# convert from stereo to mono
signal_mono_1 = torch.mean(signal_1, dim=0)
# change freq
resample_rate = 16000
resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
resampled_waveform_1 = resampler(signal_mono_1)

inputs = feature_extractor(resampled_waveform_1, padding=True, return_tensors="pt", sampling_rate=resample_rate)
print(inputs)

embeddings_1 = model(**inputs).embeddings
embeddings_1 = F.normalize(embeddings_1.squeeze(1), p=2, dim=1).detach().cpu()

# audio_path = r"role_audios\voice\クレア\0052_00.04.51.590_00.04.54.090_謝罪を求めても無駄ですわよ。.wav"
audio_path_2 = r"role_audios\voice\クレア\0050_00.04.43.240_00.04.46.950_ぼっとしていらっしゃるから置物かと思いましたわ。.wav"

signal_2, fs =torchaudio.load(audio_path_2)

# convert from stereo to mono
signal_mono_2 = torch.mean(signal_2, dim=0)
print(signal_mono.shape)

# change freq
resample_rate = 16000
resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
resampled_waveform_2 = resampler(signal_mono_2)
inputs = feature_extractor(resampled_waveform_2, padding=True, return_tensors="pt", sampling_rate=resample_rate)
embeddings_2 = model(**inputs).embeddings
embeddings_2 = F.normalize(embeddings_2.squeeze(1), p=2, dim=1).detach().cpu()
print(embeddings_2.shape)

# cosine_sim = torch.nn.CosineSimilarity(dim=-1)
print(F.cosine_similarity(embeddings_1, embeddings_2))

# audio_path = r"role_audios\voice\レイ\0007_00.02.12.060_00.02.16.730_中小企業に勤める私大橋零は.wav"
audio_path_3 = r"role_audios\voice\レイ\0023_00.03.09.380_00.03.12.550_私の名前覚えていらっしゃいますか.wav"

signal_3, fs =torchaudio.load(audio_path_3)

# convert from stereo to mono
signal_mono_3 = torch.mean(signal_3, dim=0)
print(signal_mono.shape)

# change freq
resample_rate = 16000
resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
resampled_waveform_3 = resampler(signal_mono_3)
inputs = feature_extractor(resampled_waveform_3, padding=True, return_tensors="pt", sampling_rate=resample_rate)
print(inputs, inputs["input_values"].shape)

embeddings_3 = model(**inputs).embeddings
print(embeddings_2.shape)
embeddings_3 = F.normalize(embeddings_3.squeeze(1), p=2, dim=1).detach().cpu()
print(embeddings_3.shape)

# cosine_sim = torch.nn.CosineSimilarity(dim=-1)
print(F.cosine_similarity(embeddings_1, embeddings_3))


audio_path_4 = r"role_audios\voice\レイ\0031_00.03.35.740_00.03.38.750_私はクレア様が大好きです!.wav"

signal_4, fs =torchaudio.load(audio_path_4)

# convert from stereo to mono
signal_mono_4 = torch.mean(signal_4, dim=0)
print(signal_mono.shape)

# change freq
resample_rate = 16000
resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
resampled_waveform_4 = resampler(signal_mono_4)
inputs = feature_extractor(resampled_waveform_4, padding=True, return_tensors="pt", sampling_rate=resample_rate)
print(inputs, inputs["input_values"].shape)

embeddings_4 = model(**inputs).embeddings
print(embeddings_4.shape)
embeddings_4 = F.normalize(embeddings_4.squeeze(1), p=2, dim=1).detach().cpu()
print(embeddings_4.shape)

# cosine_sim = torch.nn.CosineSimilarity(dim=-1)
print(F.cosine_similarity(embeddings_1, embeddings_4))



def extract_embeddings(audio_path: str=None):
    """Given audio paths, transforms to embeddings

    Args:
        audio_path1 (str, optional): _description_. Defaults to None.
        audio_path2 (str, optional): _description_. Defaults to None.
    """
    signal_1, fs =torchaudio.load(audio_path_1)

    # convert from stereo to mono
    signal_mono_1 = torch.mean(signal_1, dim=0)

    # change freq
    resample_rate = 16000
    resampler = T.Resample(fs, resample_rate, dtype=signal_mono.dtype)
    resampled_waveform_1 = resampler(signal_mono_1)
    inputs_1 = feature_extractor(resampled_waveform_1, padding=True, return_tensors="pt", sampling_rate=resample_rate)

    embeddings_1 = model(**inputs_1).embeddings

    return embeddings_1


# compare all the 
base_dir_clair = r"role_audios\voice\クレア"
clair = os.listdir(base_dir_clair)

embedds_clair = []
for file in clair:
    path = os.path.join(base_dir_clair, file)
    
    embedding = extract_embeddings(path)
    embedds_clair.append(embedding)
    

