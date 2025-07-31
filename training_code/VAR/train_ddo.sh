alpha=50.0 # try different alpha
beta=0.02

# change to your ImageNet path
ImageNet_path="/path/to/imagenet"

# d16, 256x256
torchrun --nproc_per_node=8 DDO_train.py --local_out_dir_path="experiments/d16/alpha${alpha}_beta${beta}" \
  --depth=16 --bs=1024 --ep=1 --tblr=1e-6 --fp16=1 --alng=1e-3 --wpe=0.1 \
  --alpha=${alpha} --beta=${beta} --ac=8 --uncond_ratio=0.5 \
  --ref_ckpt="pretrained/var_d16.pth" --data_path=${ImageNet_path}

# d30, 256x256
torchrun --nproc_per_node=8 DDO_train.py --local_out_dir_path="experiments/d30/alpha${alpha}_beta${beta}" \
  --depth=30 --bs=1024 --ep=1 --tblr=1e-6 --fp16=1 --alng=1e-3 --wpe=0.1 \
  --alpha=${alpha} --beta=${beta} --ac=32 --uncond_ratio=0.5 \
  --ref_ckpt="pretrained/var_d30.pth" --data_path=${ImageNet_path}