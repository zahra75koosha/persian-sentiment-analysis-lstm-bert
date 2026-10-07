from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")


label2id = {"HAPPY": 0, "SAD": 1}
id2label = {v: k for k, v in label2id.items()}

model.config.label2id = label2id
model.config.id2label = id2label

dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

def convert_labels(example):
    example["label"] = label2id[example["label"]]
    return example

dataset = dataset.map(convert_labels)

def preprocess_function(examples):
    return tokenizer(examples["comment"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


if os.path.exists("./trained_model"):
    print("مدل آموزش‌دیده بارگذاری شد.")
    model = AutoModelForSequenceClassification.from_pretrained("./trained_model")
else:
    print("مدل از ابتدا آموزش داده می‌شود.")
    trainer.train()
    trainer.save_model("./trained_model")


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return id2label[prediction]


while True:
    text = input("متن خود را وارد کنید (برای خروج 'exit' را تایپ کنید): ")
    if text.lower() == "exit":
        print("خروج از برنامه.")
        break
    sentiment = predict_sentiment(text)
    print(f"احساس شناسایی شده: {sentiment}")
