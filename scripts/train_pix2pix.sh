set -ex
python train.py --dataroot /mount/Users/zli/stylegan2_data/raw_images/capture_data/ --gpu_ids 3,4,5 --name completion_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 225 --dataset_mode completion --norm batch --pool_size 0 --no_flip --continue_train --display_port 8097



python train.py --dataroot /mount/Users/zli/stylegan2_data/raw_images/capture_data/ --gpu_ids 0,1,2 --name completion_pix2pix_instance --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 225 --dataset_mode completion --norm instance --pool_size 0 --no_flip --display_port 8098 --continue_train

## testing

python test.py --dataroot /mount/Users/zli/stylegan2_data/raw_images/capture_data/ --name completion_pix2pix --model pix2pix --netG unet_256 --direction AtoB --dataset_mode completion --norm batch

python test.py --dataroot /mount/Users/zli/stylegan2_data/raw_images/capture_data/ --name completion_pix2pix_instance --model pix2pix --netG unet_256 --direction AtoB --dataset_mode completion --norm instance

python test.py --dataroot /mount/Users/zli/stylegan2_data/raw_images/reduced_uv_maps/ --name completion_pix2pix_instance --model pix2pix --netG unet_256 --direction AtoB --dataset_mode completion --norm instance