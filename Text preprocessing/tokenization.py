from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
text = "They ignore our need to obtain a deep understanding of a subject, which includes memorizing and storing a richly structured database"   
print(tokenizer.tokenize(text))