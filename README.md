# Voice_DiscoGAN
キズナアイとねこますの声を入れ替える機械学習
A DiscoGAN base voice changer.

# Usage

流れ
1. 2つの.wavファイルをGANに学習させ、音声変換モデルを生成させます。
2. 変換元の音声と変換用のモデルを指定して、音声を変換させます。

手順
1. .pyファイルを格納したディレクトリに移動し、以下のコマンドを実行します。

  ``` $ python trainer.py -v (1つ目のwavファイルのパス) -w (2つ目のwavファイルのパス) ```

2. .\128_unet_64_64\preview\フォルダ内に変換された音声が出力されますので、十分と判断したら、```Ctrl-Cキー```で学習を中断させます。

3. 以下のコマンドを実行し、```wave path...```が表示されたら変換元の音声のパスを、```net path...```が表示されたら変換用のモデルのパスを入力します。

``` python convertoer.py ```



flow
1. Let the GAN learn two .Wav files and generate a voice conversion model.
2. Specify the conversion source sound and conversion model and convert the voice.

procedure
1. Change to the directory where the .py file is stored and execute the following command.

  ```$ Python trainer.py - v (path of the first wav file) - w (path of the second wav file)```

2. The converted voice will be output in the .\128_unet_64_64\preview\ folder, so if you decide it is sufficient, let ```Ctrl-C key``` interrupt the learning.

3. Execute the following command, and when ```wave path...``` is displayed, change the path of the source voice, when ``` net path ... ```is displayed, enter the path of the model for conversion.

```python convertoer.py```
