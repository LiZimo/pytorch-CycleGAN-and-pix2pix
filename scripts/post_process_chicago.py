import glob
import imageio
import cv2
import os
import ntpath
import numpy as np

input_dir = '/mount/Users/yajie/zimo_Siggraph/CFD/CFD'
output_dir = '/mount/Users/zli/stylegan2_data/raw_images/CFD_masked'
#original_dir = '/mount/Users/zli/stylegan2_data/raw_images/capture_data/test'
#original_dir = '/mount/Users/zli/stylegan2_data/raw_images/reduced_uv_maps/test'
#output_dir = '/mount/Users/zli/pix2pix_data/completion_pix2pix_instance/test_latest/post_process'
template_name = 'template_celebA_mask.png'
#template = imageio.imread(template_name)
#template = template[:,:,0:3]

template = np.array(Image.open(template_name).convert('1')).astype(np.float32)
#celebA_mask = np.clip(celebA_mask, 0., 1.)

CFD_images = glob.glob(input_dir + '/*output_uv.png')
counter = 0
for imgname in CFD_images:
	basename = ntpath.basename(imgname)
	input_img = imageio.imread(imgname)
	input_img = cv2.resize(input_img, (512,512))
	input_img = input_img[:,:,0:3]
	input_img[template == 0] = 0


	out_name = os.path.join(output_dir, basename)
	imageio.imwrite(out_name, input_img)

	print(counter)
	counter+=1