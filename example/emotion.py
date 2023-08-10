import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 텍스트 데이터와 라벨 데이터 예시
texts = ["This movie was amazing!", "I didn't like the film at all."]
labels = np.array([1, 0]) # 긍정=1, 부정=0

# 토큰화 및 시퀀스 생성
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 패딩
padded_sequences = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

# 훈련 데이터 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

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
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=128)

# 모델 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}\nAccuracy: {accuracy}")

# 새로운 텍스트로 감정 예측
new_text = ["This movie was amazing!"]
new_sequence = tokenizer.texts_to_sequences(new_text)
padded_new_sequence = pad_sequences(new_sequence, maxlen=100, padding="post", truncating="post")
prediction = model.predict(padded_new_sequence)
sentiment = "Positive" if float(prediction) >= 0.5 else "Negative"
print(f"Sentiment: {sentiment} ({float(prediction):.4f})")
