# keep track of commands for experiments

# mistral 7b 
python compute_entropy.py --output entropies_mistral7b.jsonl --model mistralai/Mistral-7B-v0.1

python visualize_entropies.py --file entropies_mistral7b.jsonl --model mistralai/Mistral-7B-v0.1 --port 7860 --host 127.0.0.1 --n 20

# qwen 2.5 7b 
python compute_entropy.py --output entropies_qwen2.5_7b.jsonl --model Qwen/Qwen2.5-7B

python visualize_entropies.py --file entropies_qwen2.5_7b.jsonl --model Qwen/Qwen2.5-7B --port 7860 --host 127.0.0.1 --n 20
