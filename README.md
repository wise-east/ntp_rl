# Next Token Prediction with Reinforcement Learning

### Overview 

This project builds on the idea from Quiet-STaR of leveraging RL to train a model to generate text for reasoning prior to predict the next token.
There are some key limitations of the original approach:
1. It is not inference-time friendly because reasoning is produced for every token. 
    - TTFT (time to first token) is high. This is addressed by Fast Quiet-STaR by teaching a model to internalize the reasoning process through curriculum learning that steadily reduces the number of reasoning tokens (16-8 $\rightarrow $ 12-4 $\rightarrow $ 8-4 $\rightarrow$) and then uses RL in the final step to completely remove any reasoning tokens.
    - Fast Quiet-STaR doesn't make any improvements to Quiet-STaR's simplistic setup of keeping a fixed reasoning budget and look-ahead token length for all tokens. We hope to address this aspect so that the model can learn to dynamically determine the reasoning budget and look-ahead token length for each token based on the context. 


### Setup 

```bash
conda create -n ntp_rl python=3.11
conda activate ntp_rl

# for quietstar
git clone git+https://github.com/wise-east/transformers.git@quietstar
cd transformers
pip install -e .
cd ..
pip install -r requirements-quietstar.txt

# for current version of huggingface transformers where fasttokenizers work
pip install -r requirements.txt
```

### Experiments 

#### Compute and visualize token entropies

Example: 
```bash
# compute entropy at each token position
python compute_entropy.py --model mistralai/Mistral-7B-v0.1

# for quietstar
python inference_quietstar.py 
```

```bash
# activate the conda environment with current version of huggingface transformers where fasttokenizers work
# visualize the entropies on a web page. darker color means the token position has higher entropy (i.e., given the text before the highlighted position, the model is more uncertain about predicting the current position).
python visualize_entropies.py --port 7860 --host 127.0.0.1 --n 20
```

### Workflow 

1. Local development on non-remote machine
2. Sync to remote machine: `./sync_ntp_rl.sh`
3. Run experiments on remote machine. 
4. Sync results back to local machine: `./sync_ntp_rl.sh -p`

The reason we do this is because cursor doesn't make code suggestions with SSH FS and USC's cluster doesn't allow Remote-SSH for direct development on the remote machine. Refer to guidelines on [USC CARC website](https://www.carc.usc.edu/user-guides/hpc-systems/endeavour/getting-started-endeavour#logging-in). 