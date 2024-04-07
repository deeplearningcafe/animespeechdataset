[Japanese/[English](README.md)]

# AnimeSpeech: アニメ字幕からの言語モデルトレーニングとテキスト音声合成のためのデータセット生成

このプロジェクトは、アニメ動画の字幕を使用して、言語モデル（LLM）のトレーニングおよびテキスト音声合成（TTS）のためのデータセット生成を容易にすることを目的としています。

## Table of Contents

1. [イントロダクション](#イントロダクション)
2. [必要条件](#必要条件)
3. [入力](#入力)
4. [機能](#機能)
    - [アノテーションの作成](#アノテーションの作成)
    - [データセットの作成](#データセットの作成)
    - [ファインチューニング LLM 対話](#ファインチューニング-llm-対話)
5. [ディレクトリ構造](#ディレクトリ構造)
6. [使用方法](#使用方法)
7. [ライセンス](#ライセンス)

## イントロダクション
AnimeSpeechは、アニメの字幕から言語モデル（LLM）のトレーニングとテキスト音声合成（TTS）のためのデータセットを生成するためのプロジェクトです。このプロジェクトは、アニメ動画の字幕を活用して、機械学習モデルのトレーニングに必要なデータを容易に作成できるように設計されています。具体的には、会話データの抽出やキャラクターの声の合成などの機能を提供し、言語モデルや音声合成の研究や開発に役立ちます。

動画からスピーカー認識は話者照合タスクと呼んでる、残念ながら今回は使えない技術です。今使ってる手法は動画から埋め込みを抽出して、`キャラ作成`の部分は字幕に人間がラベルをつけて、そのラベルされた埋め込みを基にして、KNNの学習データになります。
新しい動画を予測する時に、ラベルの埋め込みと新しい埋め込みの距離を図って、ある閾値より小さいな距離ならそのキャラとして認識します。

### 説明動画
[<img src="images\サムネイル.png" width="600" height="300"/>](https://youtu.be/2SZ8dA5AgAg)

[ブログ記事](https://aipracticecafe.site/detail/8)


## 必要条件
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


## 入力
- **ビデオファイル**: オーディオとダイアログデータを抽出するビデオ。
- **字幕ファイル**: ビデオの字幕を含む.strファイル。
- **アノテーションファイル**: 予測を含むcsvファイル。このファイルは、オーディオとダイアログのデータセットを作成するためだけに使用されます。ラベリングまたは予測には、自動的に出力する。

ビデオファイルと字幕ファイルは、`data/inputs`フォルダに配置する必要があります。アノテーションファイルの場合、フォルダのパスが必要です。例えば:
`video_name/preds.csv`が`data/outputs` フォルダにある場合です。  

## 機能
### アノテーションの作成
この機能は、字幕とビデオを処理してアノテーションを生成します。

#### キャラクターの作成
ユーザーは、所望のキャラクターのデータをラベル付けして表現を作成できます。変換された字幕は、エクセルシートのような表形式のデータに変換されます。新しいデータを予測するには、キャラクターの埋め込みが必要なので、このプロセスは少なくとも一度行う必要があります。最低限各キャラの30個ラベルを付けると思います、沢山のラベルがあればもっと精密な予測が出来ます。

今は三つモデルを使える、`SpeechBrain`、`WavLM`と`Espnet`。一番性能が高いモデルは`Espnet`、そのモデルはDockerコンテナ内で使える、詳細は[Docker](#docker)

#### キャラクターの予測
この機能は、各行を話すキャラクターを予測します。所望のキャラクターの事前作成された表現（埋め込み）が必要であり、表現を持つキャラクターについてのみ予測します。


### データセットの作成
この機能は、アノテーションファイルを入力として受け取り、LLMsとTTSのためのデータセットを作成します。

#### ダイアログデータセット
これは、LLMsのトレーニングに適した会話型データセットを作成します。ユーザーは、含めるキャラクターのダイアログを選択できます。


#### オーディオデータセット
所望のキャラクターのすべてのオーディオを抽出し、TTSトレーニング用の対応するテキストとともにフォルダに整理します。

#### ファインチューニング LLM 対話
Hugging Face Transformers ライブラリを使用して会話型言語モデル（LMs）のトレーニングを容易にするために設計されたトレーニングスクリプトです。事前学習済みモデルの読み込み、データの準備、モデルのトレーニング、およびトレーニングログの保存機能を提供します。


## ディレクトリ構造
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
├── docker
├── pretrained_models
├── src
│   ├── characterdataset
│   │   ├── api
│   │   ├── common
│   │   ├── configs
│   │   ├── datasetmanager
│   │   ├── oshifinder
│   │   ├── train_llm
├── tests
│   ├── test_dataset_manager.py
│   ├── test_finder.py
│   ├── test_train_conversational.py
├── webui_finder.py
├── train_webui.py
```
### ファイルの説明
    -data は字幕とビデオファイルを保存し、予測もそこに保存されます
    -datasetmanagerは、字幕ファイルとテキスト部分を処理するサブパッケージです。
    -oshifinderは、埋め込みを作成し、予測を行うサブパッケージです。
    -train_llmは、QLoRA手法を用いて大規模言語モデルをファインチューニングします。
    -webui_finder.pyは、Gradioベースのインタフェースです。
    -train_webui.pyは、QLoRA学習のために、Gradioベースのインタフェースです。


## 使用方法
### インストール
```bash
git clone https://github.com/deeplearningcafe/animespeechdataset
cd animespeechdataset

```
`conda`を使用する場合は、新しい環境を作成することをお勧めします。
```bash
conda create -n animespeech python=3.11
conda activate animespeech
```

その後、必要なパッケージをインストールします。もしPCにNVIDIA GPUがない場合は、要件ファイルから--index-url行を削除してください。その行はcudaソフトウェアをインストールします。
```bash
pip install -r requirements.txt
pip install -e .
```

WebUIを使用する場合は、次のコマンドを実行します。
```bash
python webui_finder.py
```

### Docker
音声埋め込みは非常に重要な部分です。その埋め込みの質によってデータセットの質も影響されます。予測の正解率が上がれば、アノテーションの修正も容易になるでしょう。ですから、可能であれば性能が最も優れているモデルを使用したいと思います。`Espnet-SPK`は対話型のデータに強力なモデルを提供しています。ただし、`Espnet`のパッケージは少し古いライブラリーを使用しています。たとえば、Python 3.10や`Torch 2.1.2`などです。逆に、`Speechbrain`や`Transformers`ライブラリは`Torch`の新しいバージョンにも対応しており、すべての環境をダウングレードする必要はないと感じました。そのため、EspnetはDockerのコンテナで使用できます。`volume`を使用して、Espnetの出力はホストのフォルダに保存されます。APIは単純な命令を実行するだけです。

Dockerを使用する場合は、次のコマンドを実行してください。
```bash
docker compose up -d
```
毎回作り直すのはもったいないため、Espnetの対話型モデルを毎回ダウンロードする必要があります。そのため、イメージとコンテナを削除したくない場合は、一時停止することができます。停止するには次のコマンドを実行します。

```bash
docker compose stop
```
再度使用する場合は、次のコマンドを実行してください。
```bash
docker compose start
```

### キャラクターの表現を作成する
1. ビデオ名と字幕名を入力します。ともに`data/inputs`に配置されています。字幕を持っていない場合は、トランスクライブチェックボックスを使用します。
2. 所望のキャラクターの表現（埋め込み）を作成します。データセットをロードするには、`load df`ボタンを使用してください。
3. ユーザーはデータフレームにラベルを付けます。最初の列にキャラクター名を入力します。
4. ラベルしたデータを保存するために、`safe annotations`ボタンを使用して
4. `Create representation`ボタンを使用して、ラベル付きデータから埋め込みを抽出します。

### キャラクターを予測する
1. ビデオ名と字幕名を入力します。ともに`data/inputs`に配置されています。字幕を持っていない場合は、トランスクライブチェックボックスを使用します。
2. `Predict characters`ボタンを使用します。作成されたアノテーションファイルのパスが`annotation file`テキストボックスに表示されます。動画と同じ名前のフォルダの中に、結果ファイルが保存されます。

### キャラクターの予測の修正
予測は完璧ではないため、アノテーションの修正が推奨されます。しかし、この作業は非常に退屈なものです。そこで、作業を少しでも楽にするために、以下の手順を用意しました。

1. 予測ファイルを`Annnotations`のテキストボックスに貼り付け、`Create file with texts and predictions`のボタンを使用します。すると `cleaning.csv`ファイルが生成されます。
2. `cleaning.csv`ファイルを使用して、音声を聞きながらテキストを修正します。
3. 修正した`cleaning.csv`ファイルを`Annnotations`のテキストボックスに貼り付け、`Update predictions`のボタンを使用します。すると `PREDICTION-FILE_cleaned.csv`ファイルが生成されます。また、埋め込みや音声ファイルの名前も変更されます。

### オーディオとダイアログデータセットを作成する
1. 予測結果ファイルを入力してください。格納されているフォルダを含めますが、`data/outputs`パートは含めません。
2. `Export for training`タブで作成するデータセットの種類、`dialogues`または`audios`を選択してください。
3. `dialogues`の場合、最初のキャラクター と 2番目のキャラクター、ユーザーロールとアシスタントロールを指定できます。`audios`の場合は、キャラクターを選択する必要があります。
4. `dialogues`の場合、2行を会話として考慮する最大時間間隔を選択できます。デフォルトは5秒です。
5. `Transform`ボタンをクリックします。

### 新しいラベル付きデータの追加
予測ファイルを修正した後、その埋め込みを学習データセットとして使用できます。新しいデータをラベル付き埋め込みデータセットに追加すると、予測の精度が向上するはずです。距離を考慮して、近隣のデータから離れたサンプルを使用したいのですが、そのようなサンプルはモデルにとって`難しい`と見なされるため、学習データとして価値が高いです。


1. 修正した予測ファイルを`Annotations`のテキストボックスに貼り付け、"Create characters"タブで、`Add new data`を展開します。
2. 最低限の距離を設定した後、0.4以上の値は疑わしいとされていますので、0.2は十分であると考えます。`Add new embeddings to the labeled data`ボタンを使用してください。埋め込みファイルは自動的に`Character embeddings`フォルダにコピーされます。


### トランスクライブ
音声認識には、reazonspeechが公開したnemoモデルを使用しています。ただし、このモジュールはWindowsで直接使用することができません。WSL2を使用する場合は問題ありません。そのため、音声認識を行うために単純なFastAPIを使用したスクリプト`asr_api.py`を含めています。

音声認識のために、Dockerイメージを作成しました。このコンテナはファイル名から音声認識の処理を行います。そして、生成された`Annnotations`ファイルはホストのディレクトリに保存されます。。`Volumes`を使ってるおかげで、コンテナはホストの`data/outputs`でファイルを保存します。APIにファイルを送信する必要がないため、プログラムの処理速度が向上します。

### KNNのために、適切なKを探す
一番良い `n_neighbors`を探すために、以下のコマンドを実行してください。:
```bash
python -m characterdataset.oshifinder.knn_choose
```


### QLoRAのトレーニング
会話型の言語モデルをトレーニングするには、トレーニングに必要なパラメーターを指定する構成ファイル（default_config.toml）が必要です。このファイルは、train_webui.py インターフェースを使用して更新することができます。トレーニングには、CMDを使用することもサポートされています。

#### 構成
```
[peft]
rank = 64
alpha = 64
dropout = 0.1
bias = "none"

[dataset]
dataset = "YOUR-DATASET-CSV-PATH"
character_name = "THE-CHARACTER-NAME-TO-LEARN"

[train]
base_model = "HUGGINGFACE-MODEL-NAME"
max_steps = 80
learning_rate = 1e-4
per_device_train_batch_size = 16
optimizer = "adamw_8bit"
save_steps = 5
logging_steps = 5
output_dir = "output"
save_total_limit = 10
push_to_hub = false
warmup_ratio = 0.05
lr_scheduler_type = "constant"
gradient_checkpointing = true
gradient_accumulation_steps = 2
max_grad_norm = 0.3
save_only_model = true
```
QLoRAをトレーニングするためには、bitsandbytes、peft などの追加のパッケージが必要です。トレーニングには、以下のコマンドを使用します。
```bash
pip install -r requirements-train.txt
```
トレーニング用の WebUI を使用するには、次のコマンドを実行してください。
```bash
python train_webui.py
```
CMD を使用して実行するには、構成ファイルへのパスを引数としてモジュールを実行します（--config_file）。`config_file` が提供されない場合は、デフォルトで`train_llm`内の構成ファイルが使用されます。
```bash
python -m characterdataset.train_llm --config_file "YOUR-CONFIG-FILE"
```

### TODO
- [ ] 場合によって、クラスのアトリビュートを関数のパラメータを変える.  
- [X] 対話データセットを作る時に、キャラの名前と(可能)を使います。
- [ ] Whisperサポートを追加する。  
- [ ] 一つのファイルだけじゃない、フォルダをすべての処理。  
- [X] QLoRAファインチューニングのスクリプトを追加する.
- [X] `n_neighbors`のパラメータを追加する.  


## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細については[LICENSE](LICENSE)ファイルを参照してください。
