import glob
import imageio
import cv2
import os
import ntpath
import numpy as np

input_dir = '/mount/Users/zli/pix2pix_data/completion_pix2pix_instance/test_latest/images'
original_dir = '/mount/Users/zli/stylegan2_data/raw_images/capture_data/test'
output_dir = '/mount/Users/zli/pix2pix_data/completion_pix2pix_instance/test_latest/post_process'
template = 'template_celebA_mask.png'

network_outputs = glob.glob(input_dir + '/*fake_B.png')
counter = 0
for imgname in network_outputs:
	basename = ntpath.basename(imgname).replace('_fake_B','')
	input_img = imageio.imread(os.path.join(original_dir, basename))
	input_img = input_img[:,:,0:3]
	network_img = imageio.imread(imgname)
	network_img = cv2.resize(network_img, (384,384), interpolation=cv2.INTER_NEAREST)
	empty_im = np.zeros((512,512,3))
	slack = 64
	empty_im[slack:-slack,slack:-slack,:] = network_img

	input_img[input_img == 0] = empty_im[input_img == 0]
	template = imageio.imread(template)[:,:,0:3]
	print(input_img.shape)
	print(template.shape)
	input_img[template == 0] = 0
	out_im = input_img.astype(np.uint8)

	out_name = os.path.join(output_dir, basename)
	imageio.imwrite(out_name, out_im)

	print(counter)
	counter+=1