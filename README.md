# Deep_VoiceChanger  
深層学習とかを使って、ボイスチェンジャー作るリポジトリ。  
A repository to make voice changer using deep learning.  

たぶん沢山の人が、仮想世界の自分の声を欲しがっていると思う。  
ここではそんな人のために、深層学習とかですごいボイチェンを作るよ。  
開発途上だから。  
I think many people wants to get own voise that use in virtual world.  
In here, we make a great voice changer using deep learning or others.  
This is now developing.  

ゆるーく作ってるので、何かあればゆるーく言ってください。  
英語が間違ってたら早めに教えて。  
I'm doing this loosely, so please tell me loosely if you find something.  
Tell me early if you find wrong English.  

## Overview  
このコードは、2人の声を入れ替える機械学習をしているよ。  
Aさんの話した内容を、Bさんの声で聴くことができるようになるよ。  
This code do machine learning that replace someone's voice to other's voice.  
You can hear contents that A spoke in B's voice.  

## Demo  
元音声A  
変換後A→B  
再変換A→B→A  

元音声B  
変換後B→A  
再変換B→A→B  
音声は[キズナアイさん](https://youtu.be/CPvD2qz-rG4?list=PL0bHKk6wuUGKbc1g6y_azaIeLwKTf1QfM&t=444)と[ねこますさん](https://youtu.be/lllCzDqlExo)です。  

## Usage  
`python trainer.py -v VOICE_FILE_PATH_A -w VOICE_FILE_PATH_B -s TEST_VOICE_FILE_PATH_A -u TEST_VOICE_FILE_PATH`  
trainer.pyを実行すると学習してくれるけど、Aさんの声のファイルとBさんの声のファイルとそれぞれのテスト用の声のファイルが必要だよ。  
**ファイルはwavの16kHz(サンプルレート)にしてね**。音量は89dBにした方がいいと思う。  
実行すると、resultsってフォルダができて、プレビューの音声が生成されるから。  
学習のIteration回数は適当だから、プレビューを聞きながら適当なところで止めてね。  
You can train by running 'trainer.py'. And you need A's voice file, B's voice file and test file.  
**File must be wav and 16kHz(sampling rate). I think 89dB of volume is good.  
When you run, a folder named 'results' will be made. And some preview voice will be made.  
Iteraton number is groundless, so you should stop train when you this preview voice is good.  

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
