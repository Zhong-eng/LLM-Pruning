## Run Code on PSC
0. ```ssh <username>@bridges2.psc.edu```
1. ```cd /ocean/projects/cis240042p/<USERNAME> && git clone ```
2. ```module load anaconda3```
3. ```conda create -n llm```
4. ```conda activate llm```
5. ```conda install pip```
6. ```pip install -r requirements.txt```
7. ```sbatch -p GPU-shared --gpus=1 submit.job```

## Useful facts
1. Check job status: ```squeue -u <username>```
2. Output is ```slurm-<id>.out```
3. Profiler output is stored in the folder ```log```. This can be viewed in tensorborad on colab. To do this, download the folder and upload to Colab. Demo is [here](https://colab.research.google.com/drive/137xvbf_m3kwHE1vzmERV25Skl4BnKFPh)
