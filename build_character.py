import os
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, Audio
import pandas as pd
import json
import random
import IPython.display as ipd
import numpy as np
vocab_train = []
vocab_test = []

df = pd.read_csv(
    "data/line_index.csv",
    header=None,
    names=["file_id", "unused", "transcription"],
    nrows=1000
)

df = df.drop(columns=["unused"])

#prefix audio path
audio_dir = os.path.normpath("./data/wavs")
df["file_id"] = df["file_id"].astype(str).str.strip()
df["path"] = df["file_id"].apply(lambda x: os.path.normpath(os.path.join(audio_dir, f"{x}.wav")))
hf_dataset = Dataset.from_pandas(df)
common_voice_train = hf_dataset
common_voice_valid = hf_dataset
# print(common_voice_train[0])

for batch in tqdm(common_voice_train):
    sentence = batch['transcription']
    vocab_train.extend(list(set(list(sentence))))

for batch in tqdm(common_voice_valid):
    sentence = batch['transcription']
    vocab_test.extend(list(set(list(sentence))))


#config unk and padding token
vocab_list = list(set(vocab_train) | set(vocab_test))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(vocab_dict)

# print(common_voice_train[0])


# common_voice_train = common_voice_train.cast_column("path", Audio(sampling_rate=16000)).rename_column('path', 'audio')
# common_voice_valid = common_voice_valid.cast_column("path", Audio(sampling_rate=16000)).rename_column('path', 'audio')


with open('./vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)