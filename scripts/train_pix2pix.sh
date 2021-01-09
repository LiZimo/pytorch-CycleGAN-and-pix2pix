set -ex
python train.py --dataroot /mount/Users/zli/stylegan2_data/raw_images/capture_data --name completion_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode completion --norm batch --pool_size 0
