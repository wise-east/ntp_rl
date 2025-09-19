# Compute entropies

## mistral 7b 
python compute_entropy.py --model mistralai/Mistral-7B-v0.1

## qwen 2.5 7b 
python compute_entropy.py --model Qwen/Qwen2.5-7B --batch-size 2

## quiet star
python inference_quietstar.py

# Visualize results
python visualize_entropies.py --port 7860 --host 127.0.0.1 --n 20