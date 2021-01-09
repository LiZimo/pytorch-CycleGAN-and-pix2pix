set -ex
python train.py --dataroot /mount/Users/zli/stylegan2_data/raw_images/capture_data/ --gpu_ids 0,1,2,3,4 --name completion_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode completion --norm batch --pool_size 0 --no_flip
