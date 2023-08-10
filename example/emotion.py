import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# 텍스트 데이터와 라벨 데이터 예시
texts = ["This movie was amazing!", "I didn't like the film at all."]
labels = np.array([1, 0]) # 긍정=1, 부정=0

# 토큰화 및 시퀀스 생성
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 패딩
padded_sequences = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

# 모델 생성
model = Sequential([
    Embedding(10000, 32, input_length=100),
    LSTM(32, dropout=0.2),
    Dense(1, activation="sigmoid")
])

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# 모델 훈련
model.fit(padded_sequences, labels, epochs=10, batch_size=128)

# 모델 저장
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
model.save("sentiment_model.h5")

# 모델 불러오기
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
model = load_model("sentiment_model.h5")

# 감정 예측 함수
def predict_sentiment(text):
    new_sequence = tokenizer.texts_to_sequences([text])
    padded_new_sequence = pad_sequences(new_sequence, maxlen=100, padding="post", truncating="post")
    prediction = model.predict(padded_new_sequence)
    sentiment = "Positive" if float(prediction) >= 0.5 else "Negative"
    return sentiment, float(prediction)

# 예측 예시
text_to_predict = "This was a great movie!"
sentiment, prediction = predict_sentiment(text_to_predict)
print(f"Text: {text_to_predict}\nSentiment: {sentiment} ({prediction:.4f})")
