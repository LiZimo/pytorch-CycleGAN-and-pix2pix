import glob
import imageio
import ntpath
import cv2

outsize = 512
#in_dir = '/mount/Users/zli/stylegan2_data/raw_images/capture_data/'
in_dir = '/vgldb1/LightStageFaceDB/Datasets/FaceEncoding/DiffuseAlbedo'
out_dir = '/mount/Users/zli/stylegan2_data/raw_images/capture_data/texture_completion_dataset'

all_files = glob.glob(in_dir + '/*.exr')
counter = 0
for imgname in all_files:
	img = imageio.imread(imgname)

	if img.shape[0] != out_size:
		out_im = cv2.resize(img, (outsize,outsize))
	else:
		out_im = img

	imageio.imwrite(out_dir + '/' + ntpath.basename(imgname), out_im)

	counter += 1
	print(counter)