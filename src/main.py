import re
import tiktoken
import tokenization

with open("assets/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print(f"Word count: {len(raw_text)}")


# Tokenization - breaking down the training set into distinct words
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# print(f"Token count: {len(preprocessed)}")

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
# print(f"Vocab size: {vocab_size}")


# Seperate training texts and add token for unknown words
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])


# Bijective mapping from tokens onto N^0 to create token IDs
vocab = {token:integer for integer,token in enumerate(all_tokens)}
# print(f"New vocab size: {len(vocab.items())}")


tokenizer = tokenization.SimpleTokenizerV1(vocab)

# Only works for words used in the-verdict
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# print(text)

# 1130: <|endoftext|>, 1131: <|unk|>
tokenizer = tokenization.SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))


# Byte pair encoding
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     " of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
