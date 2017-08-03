import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils


#params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
						   qualityLevel = 0.005,
						   minDistance = 5,
						   blockSize = 5 )
						   
# feature_params = dict( maxCorners = 500,
						   # qualityLevel = 0.05,
						   # minDistance = 5,
						   # blockSize = 12 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (10,10),
					  maxLevel = 2,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

MIN_MATCH_COUNT = 4




def get_overlap_regions2(im_left,im_right,overlap_width_l,overlap_width_r):
	hl,wl = im_left.shape[:2]

	im_left_left = im_left[:,:overlap_width_r,:]
	im_left_right = im_left[:,wl - overlap_width_l : wl,:]
	
	im_right_left = im_right[:,:overlap_width_l,:]
	im_right_right = im_right[:,wl - overlap_width_r : wl,:]
		
	return im_left_left,im_left_right,im_right_left,im_right_right
	

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

	
def draw_optical_pts(src_org,dst_org,method="good"):
	src_track = cv2.cvtColor(src_org, cv2.COLOR_BGR2GRAY)
	dst_track = cv2.cvtColor(dst_org, cv2.COLOR_BGR2GRAY)
	
	# src_clone = src_org
	# dst_clone = dst_org
	src_clone = np.copy(src_org)
	dst_clone = np.copy(dst_org)
	
	if(method == "good"):
		p_src = cv2.goodFeaturesToTrack(src_track, mask = None, **feature_params)
	elif (method == "sift"):
		sift = cv2.xfeatures2d.SIFT_create()
		kp, des = sift.detectAndCompute(src_org, None)
		p_src = np.float32([kp[idx].pt for idx in range(len(kp))]).reshape(-1, 1, 2)
	else:
		p_src = []
		
	p_dst,st = checkedTrace(src_track,dst_track,p_src) 
	good_src_pt = p_src[st == 1]
	good_dst_pt = p_dst[st == 1]
	
	
	
	corners = np.int0(p_src)
	for i in corners:
		x,y = i.ravel()
		cv2.circle(src_clone,(x,y),3,(255,255,0),-1)
		
	for i,(dst_pt,src_pt) in enumerate(zip(good_dst_pt,good_src_pt)):
		a,b = dst_pt.ravel()
		c,d = src_pt.ravel()
		cv2.circle(dst_clone,(a,b),3,(0,0,255),-1)
		cv2.circle(src_clone,(c,d),3,(0,0,255),-1)
	

	combined_im = np.concatenate((src_clone,dst_clone),axis=1)
	return combined_im,good_src_pt,good_dst_pt

	
	


def right_image_to_middle(im_right,ol_l,ol_r):
	hr,wr = im_right.shape[:2]
	result = np.zeros((hr, wr*2 ,3), np.uint8)
	wr_half = int(wr/2)
	result[:,wr_half - ol_l:wr_half+wr - ol_l,:] = im_right
	return result
#input is the orgin unwarp left image and unwarp write image
#but the coordinate of good_new and good_old are in the new image 	
def stitch_to_middle_blend(old_img,good_old,new_img,good_new,overlap_width_l,overlap_width_r):
	if len(good_new)>MIN_MATCH_COUNT:
		src_pts = np.float32([ good_new]).reshape(-1,1,2)
		dst_pts = np.float32([ good_old]).reshape(-1,1,2)


		M, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
		matchesMask = mask1.ravel().tolist()

		height,width = old_img.shape[:2]
		w_half = int(width/2)
		
		#image that focus on the left part
		right_to_warp = right_image_to_middle(new_img,overlap_width_l,overlap_width_r)
		warp_img1 = cv2.warpPerspective(right_to_warp, M, (width*2, height))
		
		
		warp_img2 = np.zeros((height, width*2,3), np.uint8)
		warp_img2[:,:int(width/2),:] = old_img[:,int(width/2):width,:]
		warp_img2[:,int(width*1.5)-(overlap_width_l+overlap_width_r):width*2-(overlap_width_l+overlap_width_r),:] = old_img[:,:int(width/2),:]
		
		im_ll,im_lr,im_rl,im_rr = get_overlap_regions2(unwarp_left,unwarp_right,overlap_width_l,overlap_width_r)
		
		x_offset_l = int(width*0.5) - overlap_width_l
		x_offset_r = int(width*1.5) - (overlap_width_l+overlap_width_r)
		mask = utils.imgLabeling2(im_lr,im_rl,im_ll,im_rr,(2560,1280),x_offset_l,x_offset_r)
		
		#fill the gap in the origin mask
		warp1_and_mask = warp_img1 + mask*255
		warp1_gap = warp1_and_mask == 0
		warp1_gap = warp1_gap.astype(int) * 1
		
		mask = mask + warp1_gap
		plt.imshow(mask),plt.show()
		labeled =  warp_img2 * mask  + warp_img1 * (1 - mask)
		
		warp1_and_mask = (1-mask)*warp_img1 + mask*255
		cv2.imwrite('video2/stitch_result/warp1_and_mask.png',warp1_and_mask)
		
		blended = utils.multi_band_blending(warp_img2, warp_img1, mask, 2)
		# cv2.imshow('p', blended.astype(np.uint8))
		# cv2.waitKey(0)
		cv2.imwrite('video2/stitch_result/stitch_result_l_'+str(overlap_width_l)+'_r'+str(overlap_width_r)+'_multi_blend.png',blended)
		
		
		# cv2.imshow('labeled',labeled.astype(np.uint8))
		# cv2.waitKey(0)
		cv2.imwrite('video2/stitch_result/stitch_result_l_'+str(overlap_width_l)+'_r'+str(overlap_width_r)+'_labeled_blend.png',labeled)
		#cv2.imwrite('video2/stitch_result/stitch_result_l_'+str(overlap_width_l)+'_r'+str(overlap_width_r)+'_blend.png',warp_img)
		

	else:
		print ("draw stitched Not enough matches are found - %d/%d" )



def stitch_to_middle(old_img,good_old,new_img,good_new,overlap_width_l,overlap_width_r):
	if len(good_new)>MIN_MATCH_COUNT:
		src_pts = np.float32([ good_new]).reshape(-1,1,2)
		dst_pts = np.float32([ good_old]).reshape(-1,1,2)


		M, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
		matchesMask = mask1.ravel().tolist()

		height,width = old_img.shape[:2]
		w_half = int(width/2)
		
		#image that focus on the left part
		warp_img = cv2.warpPerspective(new_img, M, (width*2 - (overlap_width_l+overlap_width_r), height))

			
		warp_img[:,0:w_half - overlap_width_l,:] = old_img[:,w_half:width-overlap_width_l,:]
		warp_img[:,width*2 - w_half - (overlap_width_l) :width*2 - (overlap_width_l+overlap_width_r),:] = old_img[:,overlap_width_r:w_half,:]
		
		#overlap part		
		warp_img[:,w_half-overlap_width_l:w_half - 40,:] = old_img[:,width-overlap_width_l:width-40]
		warp_img[:,int(1.5*width) - overlap_width_l - 40: int(1.5*width) - overlap_width_l,:] = old_img[:,overlap_width_r-40:overlap_width_r,:]
		
		cv2.imshow('temlate_points',warp_img)
		cv2.imwrite('video2/stitch_result/stitch_result_l_'+str(overlap_width_l)+'_r'+str(overlap_width_r)+'sift.png',warp_img)
		cv2.waitKey(0)	
		
	

	else:
		print ("draw stitched Not enough matches are found - %d/%d" )
		matchesMask = None
	
	
#retrie the images
if __name__ == "__main__":
	img = cv2.imread('video2/frame0.jpg', cv2.IMREAD_COLOR)
	im_left = img[:,:1280,:]
	im_right = img[:,1280:,:]
	
	h,w = im_left.shape[:2]
	fov = 193
	overlap_width_l = 82
	overlap_width_r = 82
	
	
	xmap,ymap = utils.buildmap_2(w,h,w,h,fov)
	unwarp_left = cv2.remap(im_left, xmap, ymap, cv2.INTER_LINEAR)
	unwarp_right = cv2.remap(im_right, xmap, ymap, cv2.INTER_LINEAR)
	

	im_ll,im_lr,im_rl,im_rr = get_overlap_regions2(unwarp_left,unwarp_right,overlap_width_l,overlap_width_r)
	
	#get optical flow pts and check the result
	combined_ll_rr,good_llrr_src,good_llrr_dst = draw_optical_pts(im_ll,im_rr)
	combined_rrll,good_rrll_src,good_rrll_dst = draw_optical_pts(im_rr,im_ll)
	

	# cv2.imwrite('video2_overlap/combined_ll_rr'+str(overlap_width_r)+'good.png',combined_ll_rr)
	# cv2.imwrite('video2_overlap/combined_rr_ll'+str(overlap_width_r)+'good.png',combined_rrll)
	
	combined_lr_rl ,good_lr_rl_src,good_lr_rl_dst = draw_optical_pts(im_lr,im_rl)
	combined_rl_lr ,good_rl_lr_src,good_rl_lr_dst = draw_optical_pts(im_rl,im_lr)

	# cv2.imwrite('video2_overlap/combined_lr_rl'+str(overlap_width_l)+'good.png',combined_lr_rl)
	# cv2.imwrite('video2_overlap/combined_rl_lr'+str(overlap_width_l)+'good.png',combined_rl_lr)
	
	
	#shift the pts
	ll_rr_shift = int(w*1.5) - (overlap_width_l + overlap_width_r) 
	lr_rl_shift = int(w*0.5) - overlap_width_l 
	
	
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
	
	right_middle = utils.right_image_to_middle(unwarp_right,overlap_width_l,overlap_width_r)
	# stitch_result,H = utils.stitch_to_middle(unwarp_left,good_left,right_middle,good_right,overlap_width_l,overlap_width_r)
	# cv2.imwrite('video2/stitch_result/stitch_result_optical_winsize_10.png',stitch_result)
	stitch_to_middle_blend(unwarp_left,good_left,unwarp_right,good_right,overlap_width_l,overlap_width_r)
	
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()




#check the stitch result