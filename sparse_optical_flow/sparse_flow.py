#!/usr/bin/python
import numpy as np
import cv2
import argparse
import utils

fov = 193
overlap_width_l = 62
overlap_width_r = 72
remap_w = 1280
remap_h = 1280
start_frame = 1
folder_name = 'result/8_10/360_0147_crop'

def main(input, output):
	cap = cv2.VideoCapture(input)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output, fourcc, 30.0, (2560 - overlap_width_l - overlap_width_r, 1280))

	xmap, ymap = utils.buildmap_2(Ws=remap_w, Hs=remap_h, Wd=1280, Hd=1280, fov=193.0)
	cap.set(1,start_frame)
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
		
		print(len(good_llrr_dst))
		print(len(good_lr_rl_dst))
		good_left = np.concatenate((good_lr_rl_src,good_rl_lr_dst,good_llrr_src,good_rrll_dst),axis = 0)
		good_right = np.concatenate((good_lr_rl_dst,good_rl_lr_src,good_llrr_dst,good_rrll_src),axis = 0)


		print(len(good_left))
		print(len(good_right))
		
		right_middle = utils.right_image_to_middle(im_r,overlap_width_l,overlap_width_r)
		stitch_result,M = utils.stitch_to_middle(im_l,good_left,right_middle,good_right,overlap_width_l,overlap_width_r)
		
		warp_gap = np.zeros((height,width*2,3))
		warp_gap[:,w_half - overlap_width_l + 4: w_half+width-overlap_width_l - 4,:] = 1
		
		warp_gap = cv2.warpPerspective(warp_gap, M, (width*2, height))
		
		#test the gap:
		warp_img2 = cv2.warpPerspective(right_middle, M, (width*2, height))
		warp_img1 = np.zeros((height, width*2,3), np.uint8)
		warp_img1[:,:int(width/2),:] = im_l[:,int(width/2):width,:]
		warp_img1[:,int(width*1.5)-(overlap_width_l+overlap_width_r):width*2-(overlap_width_l+overlap_width_r),:] = im_l[:,:int(width/2),:]
		
		
		
		cv2.imwrite(folder_name+'combined_rrll.png',combined_rrll)
		cv2.imwrite(folder_name+'combined_lr_rl.png',combined_lr_rl)
		cv2.imwrite(folder_name+'right.png',im_r)
		cv2.imwrite(folder_name+'left.png',im_l)
		
		#test_gap = warp_img2 * 0.5 + (1-warp_gap)*255
		#cv2.imshow('test_gap',test_gap.astype(np.uint8))
		
		cv2.imshow('stitch_result',stitch_result)
		cv2.imwrite(folder_name+'stitch_result_directly.png',stitch_result)
		cv2.waitKey(0)
		x_offset_l = int(width*0.5) - overlap_width_l
		x_offset_r = int(width*1.5) - (overlap_width_l+overlap_width_r) 
		
		warp_rl = warp_img2[:,w_half - overlap_width_l:w_half,:]
		warp_rr = warp_img2[:,w_half + width - overlap_width_l - overlap_width_r  :w_half + width - overlap_width_l - overlap_width_r + 60,:]
		mask = utils.imgLabeling2(im_l[:,width - overlap_width_l : width,:],warp_rl,im_l[:,:60,:],warp_rr,(width*2,height),x_offset_l,x_offset_r)
		label = mask*warp_img1 + (1-mask)*warp_img2
		warp_img2[warp_gap == 0] = warp_img1[warp_gap == 0]
		blended = utils.multi_band_blending(warp_img1, warp_img2, mask, 6)
		cv2.imwrite(folder_name+'label.png',label)
		cv2.imwrite(folder_name+'blended.png',blended)
		count = 0
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			# remap
			im_left = cv2.remap(frame[:, :1280], xmap, ymap, cv2.INTER_LINEAR)
			im_right = cv2.remap(frame[:, 1280:], xmap, ymap, cv2.INTER_LINEAR)
			right_middle = utils.right_image_to_middle(im_right,overlap_width_l,overlap_width_r)
			warp_img2 = cv2.warpPerspective(right_middle, M, (width*2, height))
			warp_img1 = np.zeros((height, width*2,3), np.uint8)
			warp_img1[:,:int(width/2),:] = im_left[:,int(width/2):width,:]
			warp_img1[:,int(width*1.5)-(overlap_width_l+overlap_width_r):width*2-(overlap_width_l+overlap_width_r),:] = im_left[:,:int(width/2),:]
			
			warp_rl = warp_img2[:,w_half - overlap_width_l + 20:w_half,:]
			warp_rr = warp_img2[:,w_half + width - overlap_width_l - overlap_width_r  :w_half + width - overlap_width_l - overlap_width_r + 60,:]
			mask = utils.imgLabeling2(im_left[:,width - overlap_width_l + 20 : width,:],warp_rl,im_left[:,:60,:],warp_rr,(width*2,height),x_offset_l,x_offset_r)
			
			label = mask*warp_img1 + (1-mask)*warp_img2
			
			#cv2.imshow('label',label.astype(np.uint8))
			# cv2.imwrite('label.png',label)
			
			
			warp_img2[warp_gap == 0] = warp_img1[warp_gap == 0]
			cv2.imwrite('mask.png',mask*255)
			blended = utils.multi_band_blending(warp_img1, warp_img2, mask, 6)
			
			print(count)
			count+= 1
			
			out.write(blended[:,:width*2 - overlap_width_l - overlap_width_r,:].astype(np.uint8))
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