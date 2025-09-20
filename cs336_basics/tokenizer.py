import regex as re
from typing import Iterator , Iterable
import json
import numpy as np
from collections import Counter
from collections import defaultdict

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                merges: list[tuple[bytes, bytes]], 
                special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        # Match longer special_tokens first to prevent shorter ones from truncating the longer ones.
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x)) 

    @classmethod
    def from_files(cls, vocab_filepath: str,
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        
        # vocab.json: {id(str): token(bytes as list of ints)}
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
            for token_id, token_bytes in vocab_json.items():
                # token_bytes 是 list[int]，需转为 bytes
                vocab[int(token_id)] = bytes(token_bytes)

        # merges.txt: 每行两个token（utf-8字符串）
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(',')
                if len(parts) == 2:
                    merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:

        text_encode: list[int] = []

        token_list = pretoken_chunk(text, self.special_tokens) #list[bytes]
        token_counter = Counter(token_list)
        token_encode = defaultdict(list[bytes])
        for token in token_counter:
            if token.decode('utf-8') in self.special_tokens:
                token_encode[token].append(self.vocab_reverse.get(token))
            else:
                token_bytes = [bytes([b]) for b in token]
                for (a, b) in self.merges:
                    token_new: list[bytes] = [] #list[bytes] 
                    i = 0
                    while i < len(token_bytes):
                        if i < len(token_bytes) - 1 and token_bytes[i] == a and token_bytes[i + 1] == b:
                            token_new.append(a + b)
                            i += 2
                        else:
                            token_new.append(token_bytes[i])
                            i += 1
                    token_bytes = token_new

                for a_bytes in token_bytes:
                    token_encode[token].append(self.vocab_reverse.get(a_bytes))

        for token in token_list:
            for i in token_encode[token]:
                text_encode.append(i)

        return text_encode
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            line_encode = self.encode(line)
            yield from line_encode

    def decode(self, ids: list[int]) -> str:
        ids_decode: str = ''
        tokens: bytes = b''

        for id in ids:
            id_bytes = self.vocab.get(id)
            tokens += id_bytes

        ids_decode = tokens.decode('utf-8', errors='replace')
        return ids_decode

def pretoken_chunk(
    text: str,
    special_tokens: list[str] | None = None
) -> list[bytes]:

    # 1. Remove special tokens by splitting the chunk at those tokens
    PAT_special_tokens = "|".join(re.escape(special_token) for special_token in special_tokens)
    #attention! 1.deal with the case when special_tokens is empty
    if PAT_special_tokens:
        PAT_special_tokens = f'({PAT_special_tokens})' # attention! 2.keep the special tokens in the result
    f_parts = re.split(PAT_special_tokens, text) if PAT_special_tokens else [text]

    # 2. Pre-tokenize 
    token_list: list[bytes] = []

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for f_part in f_parts:
        if f_part in special_tokens:
            token = f_part.encode('utf-8')
            token_list.append(token)
        else:
            tokens = [match.group(0).encode('utf-8') for match in re.finditer(PAT, f_part)]
            for token in tokens:
                token_list.append(token)

    return token_list

def decode_tinystories_to_npy(input_path):
    # 加载 vocab 和 merges
    vocab_path = 'tinystories_result/vocab.json'
    merges_path = 'tinystories_result/merges.txt'
    special_tokens = ['<|endoftext|>']

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    # 读取训练文本
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 编码
    encoded = tokenizer.encode(text)

    # 保存为.npy
    output_path = 'tinystories_result/train_encoded.npy'
    np.save(output_path, np.array(encoded, dtype=np.int32))
    print(f"编码后保存到: {output_path}, token数: {len(encoded)}")

if __name__ == '__main__':
    # input_path = 'data/TinyStoriesV2-GPT4-train.txt'
    input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
    decode_tinystories_to_npy(input_path)
