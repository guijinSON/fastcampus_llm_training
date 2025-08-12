python evaluation.py \
  --model kakaocorp/kanana-1.5-2.1b-instruct-2505 \
  --dataset KMMLU_Redux \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8192

python evaluation.py \
  --model kakaocorp/kanana-1.5-2.1b-instruct-2505 \
  --dataset MCLM \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8291

python evaluation.py \
  --model kakaocorp/kanana-1.5-2.1b-instruct-2505 \
  --dataset HRB1_0 \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8192


python evaluation.py \
  --model amphora/fc-proj1-test01 \
  --dataset KMMLU_Redux \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8192

python evaluation.py \
  --model amphora/fc-proj1-test01 \
  --dataset MCLM \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8291

python evaluation.py \
  --model amphora/fc-proj1-test01 \
  --dataset HRB1_0 \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8192

  
python evaluation.py \
  --model amphora/fc-proj1-dpo-3e-6 \
  --dataset KMMLU_Redux \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8192

python evaluation.py \
  --model amphora/fc-proj1-dpo-3e-6 \
  --dataset MCLM \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8291

python evaluation.py \
  --model amphora/fc-proj1-dpo-3e-6 \
  --dataset HRB1_0 \
  --temperature 0.7 \
  --top_p 0.9 \
  --reasoning False \
  --max_tokens 8192