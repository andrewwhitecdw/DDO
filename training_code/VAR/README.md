# DDO Training on VAR

## Preparing

- Clone the [VAR](https://github.com/FoundationVision/VAR) repository
- Follow the `Installation` section of VAR to setup the environment and download the ImageNet dataset
- Download [adm_in256_stats.npz](https://github.com/LTH14/mar/blob/main/fid_stats/adm_in256_stats.npz) from MAR and put it under `fid_stats/`
- Run `pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity` to support FID/IS computation
- Download the pretrained VAR checkpoints [var_d16.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth)/[var_d30.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d30.pth) and put them under `pretrained/`
- Put files in this folder under the VAR repo


## Training

Please refer to `train_ddo.sh`. The `--ref_ckpt` argument can be set to either the pretrained model or a finetuned model from previous DDO rounds.