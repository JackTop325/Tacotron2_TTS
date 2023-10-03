#### TEXT-TO-SPEECH WITH TACOTRON2

This repo was learned from PyTorch's Tutorial: [[Here](https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html)]

The text-to-speech pipeline:

1. Text preprocessing
* Text is encoded into a list of symbols

2. Spectrogram generation
* TacoTron2 model is used to generate a spectrogram from the encoded text

3. Time-domain conversion
* The spectrogram is then converted into a waveform. This process is called a `Vocoder`.

#### Text Processing

The pre-trained Tacotron2 model expects specific set of symbol tables.

##### 1. Character Based Encoding

* First map each character of the input text into the index of the corresponding symbol in the table.

* The set of symbols are: `_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz`

##### 2. Phoneme-based encoding

Similar to `Character based encoding`, but it uses a symbol table based on phonemes and a G2P (Grapheme-to-Phoneme) model.

#### Spectrogram Generation

TacoTron2 is used to generate the spectogram.

`torchaudio.pipelines.Tacotron2TTSBundle` bundles the matching models and processors together

#### Waveform Generation
The last process is to get the waveform from the spectrogram.

`torchaudio` provides vocoders based on `GriffinLim` and `WaveRNN`.

1. WaveRNN
2. Griffin-Lim
3. Waveglow
