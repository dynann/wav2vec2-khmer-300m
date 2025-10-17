import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor, Trainer, TrainingArguments
from dataclasses import dataclass
from typing import List, Dict, Union
import evaluate
from build_character import common_voice_valid, common_voice_train
import numpy as np
from jiwer import wer
import gc
import librosa
import os
os.environ["TORCH_AUDIO_USE_CODEC"] = "0"

data_train = common_voice_train
data_valid = common_voice_valid

processor = Wav2Vec2Processor.from_pretrained("./processor")
matric = evaluate.load("wer")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.1,
    layerdrop=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.75, 
    mask_time_length=10,
    mask_feature_prob=0.25,
    mask_feature_length=64,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("training on device", device)
model.to(device)
model.freeze_feature_encoder() 
print("⚠️  Feature extractor frozen - training transformer + classification head")
gc.collect()
torch.cuda.empty_cache()



# Prepare dataset


def prepare_dataset(batch):
    # Load audio manually with librosa
    audio_array, sampling_rate = librosa.load(batch["path"], sr=16000)
    
    # Process the audio
    batch["input_values"] = processor(
        audio_array,
        sampling_rate=sampling_rate
    ).input_values[0]
    
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    
    return batch




data_train = data_train.map(prepare_dataset, remove_columns=["file_id", "path", "transcription"])
data_valid = data_valid.map(prepare_dataset, remove_columns=["file_id", "path", "transcription"])
print("data to train ☑️☑️☑️",data_train[0]["input_values"], data_train[0]["labels"])

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Debug: Check prediction distribution
    unique, counts = np.unique(pred_ids, return_counts=True)
    print(f"Unique predicted IDs: {dict(zip(unique, counts))}")
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    
    print(f"Predictions: {pred_str[:3]}")  # First 3
    print(f"References: {label_str[:3]}")
    
    wer_score = matric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}


# Optimized training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-small-test",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=30,
    learning_rate=1e-4,              
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    eval_strategy="steps",
    max_grad_norm=1.0,
    warmup_steps=300,
    weight_decay=0.01,
    logging_first_step=True,
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)


trainer.train()
