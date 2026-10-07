from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return model.config.id2label[prediction]


while True:
    text = input("متن خود را وارد کنید (برای خروج 'exit' را تایپ کنید): ")
    if text.lower() == "exit":
        print("خروج از برنامه.")
        break
    sentiment = predict_sentiment(text)
    print(f"احساس شناسایی شده: {sentiment}")
