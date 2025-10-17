from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# Create tokenizer from vocab
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "./tokenizer",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    do_lower_case=False
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# Save processor
processor.save_pretrained("./processor")

# Verify it works
test_text = "ស្ពាន កំពង់"
encoded = tokenizer(test_text).input_ids
decoded = tokenizer.decode(encoded, skip_special_tokens=True)
print(f"Test encoding: {test_text} -> {encoded} -> {decoded}")