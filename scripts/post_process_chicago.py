import glob
import imageio
import cv2
import os
import ntpath
import numpy as np
from PIL import Image

#input_dir = '/mount/Users/yajie/zimo_Siggraph/CFD/CFD'
#input_dir = '/mount/Users/zli/stylegan2_data/raw_images/CFD_hair_masked'
#output_dir = '/mount/Users/zli/stylegan2_data/raw_images/CFD_final_output'

input_dir = '/mount/Users/yajie/zimo_Siggraph/TG_results'
output_dir = '/mount/Users/zli/stylegan2_data/raw_images/TG_mono'

if not os.path.exists(output_dir):
	os.makedir(output_dir)

#original_dir = '/mount/Users/zli/stylegan2_data/raw_images/capture_data/test'
#original_dir = '/mount/Users/zli/stylegan2_data/raw_images/reduced_uv_maps/test'
#output_dir = '/mount/Users/zli/pix2pix_data/completion_pix2pix_instance/test_latest/post_process'
template_name = 'template_celebA_mask.png'
valid_list_fname = '/mount/Users/yajie/zimo_Siggraph/CFD/CFD_validlist.txt'
valid_list = open(valid_list_fname).read()
valid_list = valid_list.split('\n')

#template = imageio.imread(template_name)
#template = template[:,:,0:3]

template = np.array(Image.open(template_name).convert('1')).astype(np.float32)
#celebA_mask = np.clip(celebA_mask, 0., 1.)

CFD_images = glob.glob(input_dir + '/*output_uv.png')
counter = 0
for imgname in CFD_images:
	basename = ntpath.basename(imgname)
	im_id =basename.replace('_output_uv.png', '.jpg')
	print(im_id)
	#if im_id not in valid_list:
		#continue

	input_img = imageio.imread(imgname)
	input_img = cv2.resize(input_img, (512,512))
	input_img = input_img[:,:,0:3]
	input_img[template == 0] = 0


	out_name = os.path.join(output_dir, basename.replace('_output_uv.png', '_output_uv_nobackground.png'))
	imageio.imwrite(out_name, input_img)

	print(counter)
	counter+=1