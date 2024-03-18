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
5. [ディレクトリ構造](#ディレクトリ構造)
6. [使用方法](#使用方法)
7. [ライセンス](#ライセンス)

## イントロダクション
AnimeSpeechは、アニメの字幕から言語モデル（LLM）のトレーニングとテキスト音声合成（TTS）のためのデータセットを生成するためのプロジェクトです。このプロジェクトは、アニメ動画の字幕を活用して、機械学習モデルのトレーニングに必要なデータを容易に作成できるように設計されています。具体的には、会話データの抽出やキャラクターの声の合成などの機能を提供し、言語モデルや音声合成の研究や開発に役立ちます。

動画からスピーカー認識は話者照合タスクと呼んでる、残念ながら今回は使えない技術です。今使ってる手法は動画から埋め込みを抽出して、`キャラ作成`の部分は字幕に人間がラベルをつけて、そのラベルされた埋め込みを基にして、KNNの学習データになります。
新しい動画を予測する時に、ラベルの埋め込みと新しい埋め込みの距離を図って、ある閾値より小さいな距離ならそのキャラとして認識します。

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


#### キャラクターの予測
この機能は、各行を話すキャラクターを予測します。所望のキャラクターの事前作成された表現（埋め込み）が必要であり、表現を持つキャラクターについてのみ予測します。


### データセットの作成
この機能は、アノテーションファイルを入力として受け取り、LLMsとTTSのためのデータセットを作成します。

#### ダイアログデータセット
これは、LLMsのトレーニングに適した会話型データセットを作成します。ユーザーは、含めるキャラクターのダイアログを選択できます。


#### オーディオデータセット
所望のキャラクターのすべてのオーディオを抽出し、TTSトレーニング用の対応するテキストとともにフォルダに整理します。


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
├── pretrained_models
├── src
│   ├── characterdataset
│   │   ├── datasetmanager
│   │   ├── oshifinder
├── webui_finder.py
```
### ファイルの説明
    -data は字幕とビデオファイルを保存し、予測もそこに保存されます
    -datasetmanagerは、字幕ファイルとテキスト部分を処理するサブパッケージです。
    -oshifinderは、埋め込みを作成し、予測を行うサブパッケージです。
    -webui_finder.pyは、Gradioベースのインタフェースです。.

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

その後、必要なパッケージをインストールします。
```bash
pip install -r requirements.txt
pip install -e .
```

WebUIを使用する場合は、次のコマンドを実行します。
```bash
python webui_finder.py
```

#### キャラクターの表現を作成する
1. ビデオ名と字幕名を入力します。ともに`data/inputs`に配置されています。字幕を持っていない場合は、トランスクライブチェックボックスを使用します。
2. 所望のキャラクターの表現（埋め込み）を作成します。データセットをロードするには、`load df`ボタンを使用してください。
3. ユーザーはデータフレームにラベルを付けます。最初の列にキャラクター名を入力します。
4. ラベルしたデータを保存するために、`safe annotations`ボタンを使用して
4. `Create representation`ボタンを使用して、ラベル付きデータから埋め込みを抽出します。

#### キャラクターを予測する
1. ビデオ名と字幕名を入力します。ともに`data/inputs`に配置されています。字幕を持っていない場合は、トランスクライブチェックボックスを使用します。
2. `Predict characters`ボタンを使用します。作成されたアノテーションファイルのパスが`annotation file`テキストボックスに表示されます。動画と同じ名前のフォルダの中に、結果ファイルが保存されます。

#### オーディオとダイアログデータセットを作成する
1. 予測結果ファイルを入力してください。格納されているフォルダを含めますが、`data/outputs`パートは含めません。
2. `Export for training`タブで作成するデータセットの種類、`dialogues`または`audios`を選択してください。
3.`dialogues`の場合、最初のキャラクター と 2番目のキャラクター、ユーザーロールとアシスタントロールを指定できます。`audios`の場合は、キャラクターを選択する必要があります。
4.`dialogues`の場合、2行を会話として考慮する最大時間間隔を選択できます。デフォルトは5秒です。
5.`Transform`ボタンをクリックします。


### トランスクライブ
音声認識には、reazonspeechが公開したnemoモデルを使用しています。ただし、このモジュールはWindowsで直接使用することができません。WSL2を使用する場合は問題ありません。そのため、音声認識を行うために単純なFastAPIを使用したスクリプト`asr_api.py`を含めています。

### TODO
- [ ] Whisperサポートを追加する。  
- [ ] 一つのファイルだけじゃない、フォルダをすべての処理。  


## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License

このプロジェクトはMITライセンスの下でライセンスされています。詳細については[LICENSE.txt](LICENSE)ファイルを参照してください。