# Deep_VoiceChanger  
深層学習とかを使って、ボイスチェンジャー作るリポジトリ。  
A repository to make voice changer using deep learning.  

たぶん沢山の人が、仮想世界の自分の声を欲しがっていると思います。  
ここではそんな人を対象に、高品質なボイスチェンジャーを作ります。  
まだ開発途上なので、いい結果ができ次第更新します。  
I think many people wants to get own voise that use in virtual world.  
In here, we make a great voice changer amed to those people.  
This is now developing, so I will update when I got good result.  

ゆるーく作ってるので、何かあればゆるーく言ってください。  
英語が間違ってたら早めに教えてください。  
I'm doing this loosely, so please tell me loosely if you find something.  
Tell me early if you find wrong English, please.  

## Overview  
技術的には、CycleGANを用いて2人の声を入れ替える機械学習をしています。  
Aさんの話した内容を、Bさんの声で聴くことができるようになります。  
This code use 'CycleGAN' to replace someone's voice to other's voice.  
You can hear contents that A spoke in B's voice.  

## Example  
[元音声A (raw)](https://github.com/pstuvwx/Deep_VoiceChanger/blob/master/demo/a.wav)  
[変換後A→B (converted A to B)](https://github.com/pstuvwx/Deep_VoiceChanger/blob/master/demo/ab.wav)  
[再変換A→B→A (reconverted A to B to A)](https://github.com/pstuvwx/Deep_VoiceChanger/blob/master/demo/aba.wav)  

[元音声B (raw)](https://github.com/pstuvwx/Deep_VoiceChanger/blob/master/demo/b.wav)  
[変換後B→A (converted B to A)](https://github.com/pstuvwx/Deep_VoiceChanger/blob/master/demo/ba.wav)  
[再変換B→A→B (reconverted B to A to B)](https://github.com/pstuvwx/Deep_VoiceChanger/blob/master/demo/bab.wav)  

デモ音声は[キズナアイさん](https://youtu.be/CPvD2qz-rG4?&t=444)と[ねこますさん](https://youtu.be/lllCzDqlExo)です。  
Example voices come from [キズナアイさん](https://youtu.be/CPvD2qz-rG4?&t=444) and [ねこますさん](https://youtu.be/lllCzDqlExo).  

**音声に関する権利は、発話者およびその所属組織にあります。  
権利者の規約に沿わない音声の利用は禁止されています。  
The right of the voices is in the speaker and the affiliation organization.  
It is prohibited to use voices that do not conform to the rules of the right holder.**  

## Usage  
`python trainer.py -v VOICE_A_FILE_PATH -w VOICE_B_FILE_PATH -s TEST_VOICE_A_FILE_PATH -u TEST_VOICE_B_FILE_PATH`  
trainer.pyを実行すると学習します。学習には、AさんとBさんの声のファイル、さらにそれぞれのテスト用の声のファイルが必要です。学習用の音声は、30分以上あったほうがいいと思います。  
テストの音声はプレビューを生成するのに使用します。テスト音声を指定しない場合は、自動的に学習用音声でプレビューを生成します。  
**ファイルはwavファイルか、プレエンコードしたnpyファイルです。** パラメーターは16kHzサンプルかつ音量は89dBとして調整したものです。プレエンコードについては、dataset.pyを確認してください。  
実行すると、resultsというフォルダができて、プレビューの音声などが生成されます。  
学習のIteration回数は適当なので、プレビューを聞きながら適当なところで止めてください。  
It can train by running 'trainer.py'. To train, you need a A's voice file, a B's voice file and both's test files. I think it is better if training voices has a length longer than 30 minutes.  
The test voices is used to make preview. If you do not specify test voice files, train voices will be used to make preview.  
**The file must be wav-file or pre-encoded npy-file.**  Code's parametor is tuned that was supposed 16kHz sampling rate and 89dB of volume. For pre-encoded npy-file, check dataset.py.  
When you run, a folder named 'results' will be made and some preview voice will be made.  
Iteraton number is groundless, so you should stop train when you think preview voice is good.  

## Requirement  
python3  
Chainer  
Cupy  
numpy  
scipy  

## LICENCE  
MIT  https://github.com/pstuvwx/Deep_VoiceChanger/blob/master/LICENSE  

一部のコードの権利は、別のライセンスが含まれています。  
Some codes contain other lisence.  
https://github.com/pfnet-research/sngan_projection/blob/master/LICENSE.md  
https://github.com/pfnet-research/chainer-gan-lib/blob/master/LICENSE  

音声に関する権利は、発話者およびその所属組織にあります。  
権利者の規約に沿わない音声の利用は禁止されています。  
The right of the voices is in the speaker and the affiliation organization.  
It is prohibited to use voices that do not conform to the rules of the right holder.
