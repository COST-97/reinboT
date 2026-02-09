<h1>ReinboT</h1>

This repo contains code for the paper:

ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning (ICML2025)
[📄 Paper](https://arxiv.org/abs/2505.07395)

```bash
# install dependencies for multi-modal goal conditioned policy
bash ./policy/install.sh
```
Download the pretrained [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) weights from Huggingface and save them in `resources/IP2P/`. 
Download the pretrained MAE encoder [mae_pretrain_vit_base.pth ](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and save it in `resources/MAE/`.
Download and unzip the [CALVIN](https://github.com/mees/calvin) dataset. 

## Training

### 1. Train model from scratch
```bash
bash policy/main_scratch.sh 1 12345 1 0.3 0.1 0 1 0.001 1 0 1 calvin 0.9 1 densed 
```
### 2. Finetune model only in successed data

```bash
bash policy/main_finetune.sh 1 12345 1 0.1 0.1 0 1 0.001 17 calvin 0.9 0 1 1
```

## Evaluation
To evaluate our model on CALVIN, you can execute the following instruction:
```bash
bash evaluate/eval_scratch.sh 1 17 1
```

## Acknowledgements

We thank the authors of the following projects for making their code and dataset open source:

- [CALVIN](https://github.com/mees/calvin)
- [GR-1](https://github.com/bytedance/GR-1)
- [GR-MG](https://arxiv.org/abs/2408.14368)

## Citation

If you find this project useful, please star the repository and cite our paper:
```
@article{zhang2025reinbot,
  title={ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning},
  author={Zhang, Hongyin and Zhuang, Zifeng and Zhao, Han and Ding, Pengxiang and Lu, Hongchao and Wang, Donglin},
  journal={arXiv preprint arXiv:2505.07395},
  year={2025}
}
```
