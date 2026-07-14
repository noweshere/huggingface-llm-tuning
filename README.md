# Spotify Playlist AI

 A small LLM fine-tuned from the TinyLlama-1.1B-Chat-v1.0 model from HuggingFace that was trained on the maharshipandya/spotify-tracks-dataset from HuggingFace. After training, the model should be able to create custom playlists with data from Spotify based on the user's preferences.
 The main fine-tuning.py script can be modified easily to accomidate other models and datasets. The datasets can be loaded locally or through HuggingFace by changing lines 75, 77, and 86. Line 75 chooses to pull either from HuggingFace or locally, line 77 is the name of the dataset in HuggingFace, and line 86 is the path to the local dataset.
 As for models, the fine-tuning.py script can be modified on lines 61 and 64 to accomidate other models. Line 61 is the name of the model on HuggingFace, and line 64 defines the tokenizer. For most models, you will want to leave line 64 as None to use the default tokenizer for that model.

 You can find the fine-tuned model from this project on HuggingFace:
 (https://huggingface.co/noweshere)

![add image descrition here](direct image link here)

## The Algorithm

#### Pipeline:
      1. Validate configuration.
      2. Set random seeds for reproducibility.
      3. Load and configure the tokenizer.
      4. Load the base model.
      5. Apply LoRA adapters.
      6. Load, format, and tokenize the dataset.
      7. Build the Trainer.
      8. Train with OOM recovery.
      9. Save results.
      10. Final summary.

## Running this project

#### Note: In order to run this project, you need Jetpack 6.2 or newer.

### Starting Training
1. Clone this repo:
```bash
git clone https://github.com/noweshere/huggingface-llm-tuning.git

# Go into the folder
cd huggingface-llm-tuning

# Initialize
git init .
```
2. Install dependencies:
```bash
python3 -m pip install --upgrade pip && \
python3 -m pip install \
    transformers==4.38.2 \
    numpy==1.26.4 \
    tensorboard==2.21.0 \
    "peft>=0.9,<0.12" \
    "accelerate>=0.27,<0.30"
```
3. (Optional) Change any training settings:
At the top of the fine-tuning.py script, there aree variables to change the dataset and model used, as well as other configuration settings.
4. Run the script:
```bash
python3 fine-tuning.py
```

### Running the Model
1. At the top of the fine-tuning.py script, change RUN_MODE on line 34 to be "inference".
2. Run the script:
```bash
python3 fine-tuning.py
```

[View a video explanation here](video link)
