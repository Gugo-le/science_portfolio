import tensorflow as tf
import numpy as np

# 데이터 예시
texts = ["This movie was amazing!", "I didn't like the film at all."]
labels = np.array([1, 0]) # 긍정=1, 부정=0

# 토큰화 및 패딩 함수
def tokenize_and_pad(texts, max_len=100):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return padded_sequences, tokenizer

# 모델 학습 함수
def train_model(texts, labels, max_len=100, batch_size=64, epochs=50):
    padded_sequences, tokenizer = tokenize_and_pad(texts, max_len=max_len)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=max_len),
        tf.keras.layers.LSTM(32, dropout=0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(padded_sequences, labels, batch_size=batch_size, epochs=epochs)
    return model, tokenizer

# 모델 예측 함수
def predict_sentiment(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=model.input_shape[1], padding="post", truncating="post")
    prediction = model.predict(padded_sequence)[0, 0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment, prediction

# 모델 학습 및 예측 예시
model, tokenizer = train_model(texts, labels)
text_to_predict = "I like you😍"
sentiment, prediction = predict_sentiment(model, tokenizer, text_to_predict)
print(f"Text: {text_to_predict}\nSentiment: {sentiment} ({prediction:.4f})")
