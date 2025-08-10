#!/usr/bin/env python

# python evaluation.py \
#   --model kakaocorp/kanana-1.5-8b-instruct-2505 \
#   --dataset ArenaHard \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --reasoning False \
#   --max_tokens 1024

import argparse
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
# from eval_utils import system_prompt, safe_parse
# from litellm import batch_completion
import os
os.environ['OPENAI_API_KEY'] = ""

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a vLLM model on a dataset and save its responses.")
    p.add_argument("--model", required=True,
                   help="Hugging Face model ID or local path")
    p.add_argument("--dataset", required=True,
                   help="Dataset ID or local path")
    p.add_argument("--split", default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--revision", default=None,
                   help="Model revision / commit hash (optional)")
    p.add_argument("--max_tokens", type=int, default=32768,
                   help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Top-p / nucleus sampling (default: 1.0 = disabled)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (auto-generated if omitted)")
    p.add_argument("--reasoning", type=bool, default=True, 
                   help="Whether the model uses system 2 thinking.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- build LLM ----------------------------------------------------------
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": torch.cuda.device_count(),
    }
    if args.revision:
        llm_kwargs["revision"] = args.revision

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # ---- load data ----------------------------------------------------------
    df = load_dataset('HAERAE-HUB/KoSimpleEval',args.dataset, split=args.split).to_pandas()

    # ---- craft prompts ------------------------------------------------------
    prompts = []
    
    if args.dataset in ['ArenaHard']:
        for _, row in df.iterrows():
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row['prompts']}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
    else:
        for _, row in df.iterrows():
            query = (
                f"{row['question']}\n"
                "문제 풀이를 마친 후, 최종 정답을 다음 형식으로 작성해 주세요: \\boxed{{N}}."
            )
            # query = (
            #     "다음은 {row.Category}에 관한 객관식 문제입니다. 당신의 추론 과정을 간결하게 요약한 후, ”따라서, 정답은: X”라고 결론지으십시오. 여기서 X는 A, B, C, D 중 하나입니다.\n"
            #     f"{row['question']}"
            # )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
    
    # ---- generate -----------------------------------------------------------
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling)
    df["response"] = [o.outputs[0].text for o in outputs]

    # ---- evaluate -----------------------------------------------------------
    if args.dataset in ['ArenaHard']:
        print('Starting Evaluation...')
        qrys = []
        for _,row in df.iterrows():
            response = row.response
            
            if args.reasoning:
                try:
                    response = row.response.split('</think>')[-1].strip()
                except:
                    response = "The model failed to return a response."
                
            query = [{'role':'system','content':system_prompt}]
            content = (
                "### Response A\n\n:"
                f"{row.ref}\n\n"
                "### Response B\n\n:"
                f"{response}\n\n"
            )
            query.append(
                {'role':'user','content':content}
            )
            qrys.append(query)
        responses = batch_completion(model='gpt-4.1',messages = qrys)
        df['judge'] = [safe_parse(res) for res in responses]
    
    # ---- save ---------------------------------------------------------------
    if args.output is None:
        safe_model = args.model.replace("/", "_")
        safe_data = args.dataset.replace("/", "_")
        args.output = f"{safe_data}--{safe_model}.csv"

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows ➜ {args.output}")


if __name__ == "__main__":
    main()


## KMMLU-Redux Evaluation
# import pandas as pd 
# from math_verify import parse

# df = pd.read_csv('amphora_kmmlu-redux-amphora_ko-r1-3.0.1-1.csv')

# output = []
# for _,row in df.iterrows():
#     try:
#         response = row.response.split('</think>')[1]
#         resp = parse(response)
#         output.append(resp[-1])
#     except:
#         output.append('-1')
# df['pred'] = output     


# # 1) Define mappings
# num2letter = {1: "A", 2: "B", 3: "C", 4: "D", 5:'E'}
# letter2num = {v: k for k, v in num2letter.items()}

# # 2) Normalize both sides to numbers (1–4) in a new column pred_num
# def to_pred_num(x):
#     # if it’s already an integer or a numeric string
#     if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()):
#         try:
#             return int(x)
#         except ValueError:
#             pass
#     # if it’s one of the letters A–D
#     if isinstance(x, str) and x.upper() in letter2num:
#         return letter2num[x.upper()]
#     # otherwise
#     return None

# df["pred_num"] = df["pred"].apply(to_pred_num)

# # 3) Optionally also create gold_letter if you want letter view
# df["gold_letter"] = df["gold"].map(num2letter)

# # 4) Now mark correctness
# df["correct"] = df["pred_num"] == df["gold"]

# # 5) Overall accuracy
# overall_acc = df["correct"].mean()
# print(f"Overall accuracy: {overall_acc:.2%}")

# # 6) Accuracy by category
# acc_by_cat = df.groupby("category")["correct"].mean().sort_values(ascending=False)
# print("\nAccuracy by category:")
# print(acc_by_cat)


### Math Evaluation
# import pandas as pd 
# from math_verify import parse, verify

# df = pd.read_csv('MCLM-kakaocorp_kanana-1.5-8b-instruct-2505.csv')

# output = []
# for _,row in df.iterrows():
#     try:
#         response = row.response#.split('</think>')[1]
#         resp = parse(response)
        
#         is_correct0 = verify(parse(row.gold),resp)
#         is_correct1 = row.gold == resp[-1]

#         output.append(
#             any([is_correct0,is_correct1])
#         )
#     except:
#         output.append(False)
# df['correct'] = output     

# overall_acc = df["correct"].mean()
# print(f"Overall accuracy: {overall_acc:.2%}")

# # 6) Accuracy by category
# acc_by_cat = df.groupby("category")["correct"].mean().sort_values(ascending=False)
# print("\nAccuracy by category:")
# print(acc_by_cat)



### Chat
# import pandas as pd

# # load
# df = pd.read_csv('ArenaHard-kakaocorp_kanana-1.5-8b-instruct-2505.csv')

# # extract and map judge → score (from before)
# pattern = r'(A>>B|A>B|A=B|A<B|A<<B|B>>A|B>A|B=A|B<A|B<<A)'
# score_map = {
#     'A>>B': 0, 'A>B': 2, 'A=B': 3, 'A<B': 4, 'A<<B': 5,
#     'B>>A': 5, 'B>A': 4, 'B=A': 3, 'B<A': 2, 'B<<A': 1,
# }
# df['relation'] = df['judge'].str.extract(pattern, expand=False)
# df['score']    = df['relation'].map(score_map)

# # compute response length
# #  • character‐level:
# df['resp_len_chars']  = df['response'].str.len()
# #  • token‐level (words split on whitespace):
# df['resp_len_tokens'] = df['response'].str.split().str.len()

# # points per 100 tokens
# df['eff_per_100_tokens'] = df['score'] / df['resp_len_tokens'] * 100

# # points per 1 000 characters
# df['eff_per_1k_chars']   = df['score'] / df['resp_len_chars'] * 1000

# total_score = df['score'].mean(skipna=True)
# n_errors    = df['score'].isna().sum()

# print(f"Score = {total_score},  Errors = {n_errors}")
# print("Avg points/100 tokens: ", df['eff_per_100_tokens'].mean())
# print("Avg points/1k chars:   ", df['eff_per_1k_chars'].mean())
