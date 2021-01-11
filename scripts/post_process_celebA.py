import glob
import imageio
import cv2
import os
import ntpath

input_dir = '/mount/Users/zli/pix2pix_data/completion_pix2pix_instance/test_latest/images'
output_dir = '/mount/Users/zli/pix2pix_data/completion_pix2pix_instance/test_latest/post_process'

network_outputs = glob.glob(input_dir + '/*fake_B.png')

for imgname in network_outputs:
	input_img = imageio.imread(imgname.replace('fake', 'real'))
	network_img = imageio.imread(imgname)

	input_img[input_img == 0] = network_img[input_img == 0]
	slack = 64
	empty_im = np.zeros((512,512,3))
	resized = cv2.resize(input_img, (384,384), interpolation=cv2.INTER_NEAREST)
	empty_im[slack:-slack,slack:-slack,:] = resized

	out_name = os.path.join(output_dir, ntpath.basename(imgname).replace('_fake_B',''))
	imageio.imwrite(empty_im, )