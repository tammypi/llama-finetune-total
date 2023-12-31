import argparse
import json
import numpy as np
import random
import tqdm.auto as tqdm

import datasets
import transformers


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str,default='/root/autodl-tmp/models/tokenizer.model')
    parser.add_argument("--jsonl_path", type=str,default='/root/autodl-tmp/trainData/instinwild_ch.json')
    parser.add_argument("--save_path", type=str,default='/root/autodl-tmp/UMLSE_TOKENIZED')
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)
    all_tokenized = []
    data = json.loads(open(args.jsonl_path, "r").read())
    for elem in data:
        instruction = "Question: " + elem["instruction"]
        sentence = instruction + " Answer: " + elem["output"]
        all_tokenized.append(tokenizer.encode(sentence))
    random.shuffle(all_tokenized)

    all_tokens = [1] + [
        tok
        for row in all_tokenized
        for tok in row + [tokenizer.eos_token_id, tokenizer.bos_token_id]
    ]

    truncated_tokens = all_tokens[:(len(all_tokens) // args.max_seq_length) * args.max_seq_length]
    arr = np.array(truncated_tokens).reshape(-1, args.max_seq_length)
    ds = datasets.Dataset.from_dict({"input_ids": arr})
    ds.save_to_disk(args.save_path)
    print(f"Generated {arr.shape[0]} samples.")


if __name__ == "__main__":
    main()
