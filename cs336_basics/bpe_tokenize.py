from multiprocessing import Pool
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
import time
import os
import numpy as np
import json
from collections import Counter
import csv

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    #part1 init
    start_time = time.time()

    #vocab init
    vocab = {i : bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    #merges init
    merges: list[tuple[bytes, bytes]] = []

    end_time = time.time()
    print(f"初始化耗时: {end_time - start_time:.2f} 秒")

    #part2 pre-tokenization
    start_time = time.time()

    #multi processes pre-token
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')

    task_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        process_results = pool.map(task_chunk, task_args)

    token_counter = Counter()
    for process_result in process_results:
        token_counter.update(process_result)

    end_time = time.time()
    print(f"预分词耗时: {end_time - start_time:.2f} 秒")

    #part3 BPE merges
    start_time = time.time()

    #compute BPE merges
    #(1). get pair_counts and pair_tokenidset
    pair_counts = defaultdict(int)
    pair_tokenset = defaultdict(set)  #when merge pair just change the token in tokenset
    token_bytes = defaultdict(list[bytes])
    for token, count in token_counter.items():
        token_bytes[token] = [bytes([b]) for b in token]
        token_list = token_bytes[token]
        for i in range(len(token_list) - 1):
            pair = (token_list[i], token_list[i + 1])
            pair_counts[pair] += count
            pair_tokenset[pair].add(token)

    while len(vocab) < vocab_size:
        if not pair_counts:        # no more pairs to merge
            break

        #(2). get max_count_pair and max_count
        max_count_pair: tuple[bytes, bytes] = None
        max_count = -1
        for pair, count in pair_counts.items():
            if count > max_count:
                max_count_pair = pair
                max_count = count
            elif count == max_count:
                if pair > max_count_pair:
                    max_count_pair = pair

        merges.append(max_count_pair)
        a, b = max_count_pair
        new_token = a + b
        vocab[len(vocab)] = new_token

        affected_tokenset = pair_tokenset[max_count_pair].copy()
        for token in affected_tokenset:
            token_list = token_bytes[token]
            num_token = token_counter[token]
            for j in range(len(token_list) - 1):
                old_pair = (token_list[j], token_list[j + 1])
                pair_counts[old_pair] -= num_token
                pair_tokenset[old_pair].discard(token)
                if pair_counts[old_pair] == 0:
                    pair_counts.pop(old_pair)
                    pair_tokenset.pop(old_pair)
        
            token_after_merge = []
            j = 0
            while j < len(token_list):
                if j < len(token_list) - 1 and token_list[j] == a and token_list[j + 1] == b:
                    token_after_merge.append(new_token)
                    j += 2
                else:
                    token_after_merge.append(token_list[j])
                    j += 1
            token_bytes[token] = token_after_merge

            token_list = token_bytes[token]
            for j in range(len(token_list) - 1):
                pair = (token_list[j], token_list[j + 1])
                pair_counts[pair] += num_token
                pair_tokenset[pair].add(token)

    end_time = time.time()
    print(f"BPE合并耗时: {end_time - start_time:.2f} 秒, 词表大小: {len(vocab)}, BPE合并次数: {len(merges)}")
    
    return vocab, merges

    
def task_chunk(
    task_args: tuple[str, int, int, list[str]]
) -> Counter:
    #Counter {token: count} token: bytes, count: int
    input_path, start, end, special_tokens= task_args

    with open(input_path, 'rb') as f:
        f.seek(start)
        task_f = f.read(end - start).decode('utf-8', errors = 'ignore')

    # 1. Remove special tokens by splitting the chunk at those tokens
    PAT_special_tokens = "|".join(re.escape(special_token) for special_token in special_tokens)
    f_parts = re.split(PAT_special_tokens, task_f)

    # 2. Pre-tokenize and token-count
    token_list: list[bytes] = []

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for f_part in f_parts:
        tokens = [match.group(0).encode('utf-8') for match in re.finditer(PAT, f_part)]
        for token in tokens:
            token_bytes = token
            token_list.append(token_bytes)

    return Counter(token_list)


def train_bpe_tinystories():

    input_path = 'data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    num_processes = 8

    start_time = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes)
    end_time = time.time()
    duration = end_time - start_time

    # 序列化 vocab 为 JSON，merges 为 TXT
    output_path = 'tinystories_result'
    os.makedirs(output_path, exist_ok=True)
    # vocab: {int: bytes} 存为 {str: list[int]} 格式，兼容 json
    vocab_json = {str(k): list(v) for k, v in vocab.items()}
    with open(os.path.join(output_path, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    # merges: 每行两个token（utf-8字符串）
    with open(os.path.join(output_path, 'merges.txt'), 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', errors='replace')},{b.decode('utf-8', errors='replace')}\n")

    print(f"训练耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")

    # 输出最长 token(实际上byte_pair)
    max_token_id, max_token = max(vocab.items(), key=lambda x: len(x[1]))
    print(f"最长 token id: {max_token_id}, 长度: {len(max_token)}, 内容: {max_token}")



def train_bpe_expts_owt():

    input_path = 'data/owt_train.txt'
    vocab_size = 32000
    special_tokens = ['<|endoftext|>']
    num_processes = 8

    start_time = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes)
    end_time = time.time()
    duration = end_time - start_time

    # 序列化 vocab 为 JSON，merges 为 TXT
    output_path = 'owt_result'
    os.makedirs(output_path, exist_ok=True)
    vocab_json = {str(k): v.decode('utf-8', errors='replace') for k, v in vocab.items()}
    with open(os.path.join(output_path, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_path, 'merges.txt'), 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")

    print(f"训练耗时: {duration:.2f} 秒 ({duration/3600:.2f} 小时)")

    # 输出最长 token(实际上byte_pair)
    max_token_id, max_token = max(vocab.items(), key=lambda x: len(x[1]))
    print(f"最长 token id: {max_token_id}, 长度: {len(max_token)}, 内容: {max_token}")


if __name__ == "__main__":
    train_bpe_tinystories()
    # train_bpe_expts_owt()

