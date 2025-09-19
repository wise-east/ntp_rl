# Compute entropies

## mistral 7b 
conda activate ntp_rl_vis
python compute_entropy.py --model mistralai/Mistral-7B-v0.1
python compute_entropy.py --model mistralai/Mistral-7B-v0.1 --dataset open-web-math/open-web-math --split train --text-field text --subset ""


## qwen 2.5 7b 
python compute_entropy.py --model Qwen/Qwen2.5-7B --batch-size 1 # reduced batch size so that it fits in memory for a v100

## quiet star
conda activate ntp_rl
python inference_quietstar.py # allenai/c4 
python inference_quietstar.py --dataset open-web-math/open-web-math --split train --text-field text --subset ""


# Visualize results
python visualize_entropies.py --port 7860 --host 127.0.0.1 --n 20