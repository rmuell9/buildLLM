import re

with open("assets/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Word count: {len(raw_text)}")


# Tokenization - breaking down the training set into distinct words
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(f"Token count: {len(preprocessed)}")

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(f"Vocab size: {vocab_size}")


# Bijective mapping from tokens onto N^0 to create token IDs
vocab = {token:integer for integer,token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    # input text -> token IDs
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # token IDs -> input text
    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
# Only works for words used in the-verdict
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
