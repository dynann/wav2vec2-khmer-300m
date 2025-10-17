from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file='./vocab.json',
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
tokenizer.save_pretrained('./tokenizer')
