#!/usr/bin/python
import numpy as np
import cv2
import argparse
import utils

fov = 193
overlap_width_l = 82
overlap_width_r = 82


def main(input, output):
	cap = cv2.VideoCapture(input)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output, fourcc, 30.0, (2560, 1280))

	xmap, ymap = utils.buildmap_2(Ws=1280, Hs=1280, Wd=1280, Hd=1280, fov=193.0)

	ret, frame = cap.read()
    
	#find H in first frame
	if ret == True:
		im_l = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
		im_r = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)
		
		height,width = im_l.shape[:2]
		w_half = int(width/2)
		
		im_ll,im_lr,im_rl,im_rr = utils.get_overlap_regions2(im_l,im_r,overlap_width_l,overlap_width_r)
		combined_ll_rr,good_llrr_src,good_llrr_dst = utils.draw_optical_pts(im_ll,im_rr)
		combined_rrll,good_rrll_src,good_rrll_dst = utils.draw_optical_pts(im_rr,im_ll)

		combined_lr_rl ,good_lr_rl_src,good_lr_rl_dst = utils.draw_optical_pts(im_lr,im_rl)
		combined_rl_lr ,good_rl_lr_src,good_rl_lr_dst = utils.draw_optical_pts(im_rl,im_lr)

	
		#shift the pts
		ll_rr_shift = int(width*1.5) - (overlap_width_l + overlap_width_r) 
		lr_rl_shift = int(width*0.5) - overlap_width_l 
		
		ll_rr_shift_arr = np.array((ll_rr_shift,0))
		lr_rl_shift_arr = np.array((lr_rl_shift,0))
		
		good_llrr_dst = good_llrr_dst+ll_rr_shift_arr
		good_llrr_src = good_llrr_src+ll_rr_shift_arr
		good_rrll_src = good_rrll_src+ll_rr_shift_arr
		good_rrll_dst = good_rrll_dst+ll_rr_shift_arr

		good_lr_rl_dst = good_lr_rl_dst+lr_rl_shift_arr
		good_lr_rl_src = good_lr_rl_src+lr_rl_shift_arr
		good_rl_lr_src = good_rl_lr_src+lr_rl_shift_arr
		good_rl_lr_dst = good_rl_lr_dst+lr_rl_shift_arr
		

		good_left = np.concatenate((good_lr_rl_src,good_rl_lr_dst,good_llrr_src,good_rrll_dst),axis = 0)
		good_right = np.concatenate((good_lr_rl_dst,good_rl_lr_src,good_llrr_dst,good_rrll_src),axis = 0)


		print(len(good_left))
		print(len(good_right))
		
		right_middle = utils.right_image_to_middle(im_r,overlap_width_l,overlap_width_r)
		stitch_result,M = utils.stitch_to_middle(im_l,good_left,right_middle,good_right,overlap_width_l,overlap_width_r)
		cv2.imshow('stitch_result',stitch_result)
		cv2.waitKey(0)
		
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			# remap
			im_left = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
			im_right = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)
			right_middle = utils.right_image_to_middle(im_right,overlap_width_l,overlap_width_r)
			warp_img = cv2.warpPerspective(right_middle, M, (2560, 1280))
			warp_img[:,0:w_half - overlap_width_l,:] = im_left[:,w_half:width-overlap_width_l,:]
			warp_img[:,width*2 - w_half - (overlap_width_l) :width*2 - (overlap_width_l+overlap_width_r),:] = im_left[:,overlap_width_r:w_half,:]
		
			#overlap part		
			warp_img[:,w_half-overlap_width_l:w_half - 40,:] = im_left[:,width-overlap_width_l:width-40]
			warp_img[:,int(1.5*width) - overlap_width_l - 40: int(1.5*width) - overlap_width_l,:] = im_left[:,overlap_width_r-40:overlap_width_r,:]

			out.write(warp_img.astype(np.uint8))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
			
	# Release everything if job is finished
	cap.release()
	out.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="A summer research project to seamlessly stitch dual-fisheye video into 360-degree videos")
    ap.add_argument('input', metavar='INPUT.XYZ',
                    help="path to the input dual fisheye video")
    ap.add_argument('-o', '--output', metavar='OUTPUT.XYZ', required=False, default='output.MP4',
                    help="path to the output equirectangular video")

    args = vars(ap.parse_args())
    main(args['input'], args['output'])