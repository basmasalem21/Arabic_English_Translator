🌐 Arabic ↔ English Neural Translator (Fine-Tuning)

This project implements a Neural Machine Translation (NMT) system for translating between Arabic and English using transformer-based models from Hugging Face.

The model is fine-tuned on the OPUS-100 dataset (ar-en) using Helsinki-NLP/opus-mt-ar-en.

📌 Project Goal

Build a sequence-to-sequence model that can:

🇪🇬Arabic → English translation
🇬🇧 English → Arabic translation (optional extension)
📊 Dataset

We use:

👉 OPUS-100 (ar-en) from Hugging Face Datasets

from datasets import load_dataset

ds = load_dataset("opus100", "ar-en")

Dataset structure:

translation:
    - ar: Arabic sentence
    - en: English sentence
🧠 Model

We use:

👉 Helsinki-NLP/opus-mt-ar-en

A pretrained Transformer model based on MarianMT.

⚙️ Installation
git clone https://github.com/your-username/Arabic_English_Translator.git
cd Arabic_English_Translator

pip install -r requirements.txt
🔄 Data Preprocessing

We tokenize Arabic-English pairs into model inputs:

def preprocess(example):
    src = example["translation"]["ar"]
    tgt = example["translation"]["en"]

    model_inputs = tokenizer(src, truncation=True)
    labels = tokenizer(text_target=tgt, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
🚀 Training

We fine-tune using Hugging Face Trainer:

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)
🏋️ Model Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
🔮 Inference (Translation)
Arabic → English
text = "أنا أحب الموز ولكنني أكره البرتقال"

inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
💾 Save Model
trainer.save_model("./my_model")
tokenizer.save_pretrained("./my_model")
📈 Improvements
Use facebook/nllb-200 for better accuracy
Add BLEU score evaluation
Add English → Arabic direction
Build Streamlit / FastAPI UI
Use dynamic padding for faster training
🧠 Tech Stack
Python 🐍
Hugging Face Transformers 🤗
Datasets Library 📊
PyTorch 🔥
📌 Notes
OPUS dataset is large → you can sample for faster training
Always remove unused columns during preprocessing
Use GPU for training (recommended)
🚀 Example Output
Input: أنا أحب الموز ولكنني أكره البرتقال  
Output: I love bananas but I hate oranges
📜 License

This project is for educational and research purposes.
