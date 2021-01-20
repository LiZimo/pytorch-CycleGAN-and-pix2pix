import glob
import imageio
import cv2
import os
import ntpath
import numpy as np
from PIL import Image

input_dir = '/mount/Users/yajie/zimo_Siggraph/celeba-1024_neutral_results'
output_dir = '/mount/Users/zli/stylegan2_data/raw_images/celebA_highqual_masked'
output_reverse_hair_dir = '/mount/Users/zli/stylegan2_data/raw_images/hair_masks_reverse'
#original_dir = '/mount/Users/zli/stylegan2_data/raw_images/capture_data/test'
#original_dir = '/mount/Users/zli/stylegan2_data/raw_images/reduced_uv_maps/test'
#output_dir = '/mount/Users/zli/pix2pix_data/completion_pix2pix_instance/test_latest/post_process'
hair_mask_dir = '/mount/Users/zli/stylegan2_data/raw_images/hair_masks'
template_name = 'template_celebA_mask.png'
valid_list_fname = '/mount/Users/yajie/zimo_Siggraph/CFD/CFD_validlist.txt'
valid_list = open(valid_list_fname).read()
valid_list = valid_list.split('\n')

#template = imageio.imread(template_name)
#template = template[:,:,0:3]

template = np.array(Image.open(template_name).convert('1')).astype(np.float32)
#celebA_mask = np.clip(celebA_mask, 0., 1.)

images = glob.glob(input_dir + '/*output_uv.png')
counter = 0
for imgname in images:
	basename = ntpath.basename(imgname)
	out_name = os.path.join(output_dir, basename.replace('_output_uv.png', '_output_uv_nobackground.png'))

	if os.path.exists(out_name):
		print('done')
		continue

	# if counter < 1000:
	# 	counter += 1
	# 	print('finished previously, skipping')
	# 	continue
	

	input_img = imageio.imread(imgname)
	input_img = cv2.resize(input_img, (512,512))
	input_img = input_img[:,:,0:3]

	hair_mask_name = hair_mask_dir + '/' + basename
	# if os.path.exists(hair_mask_name):
	input_hair_mask = np.array(Image.open(hair_mask_name).convert('1')).astype(np.float32)
	input_hair_mask = cv2.resize(input_hair_mask, (512,512))
	# else:
	# 	input_hair_mask = np.zeros(template.shape)
	reverse_hair_mask = 1. - input_hair_mask
	input_img[template == 0] = 0
	input_img[reverse_hair_mask == 0] = 0



	
	imageio.imwrite(out_name, input_img)
	imageio.imwrite(output_reverse_hair_dir + '/' + basename, Image.fromarray(np.uint8(reverse_hair_mask*255.))
)

	print(counter)
	counter+=1