import nltk

# nltk 데이터 경로 설정
nltk.data.path.append("/usr/local/lib/nltk_data")

from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문 장 토 큰 화1:', sent_tokenize(text))
