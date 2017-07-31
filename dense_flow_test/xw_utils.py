import cv2
import numpy as np
from matplotlib import pyplot as plt


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
lk_params = dict( winSize  = (5,5),
					  maxLevel = 2,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

MIN_MATCH_COUNT = 4


def get_theta_phi_2(x_proj, y_proj, W, H, fov):
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * np.pi / H

    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = np.cos(theta_alt) * np.cos(phi_alt)

    return np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)


def buildmap_2(Ws, Hs, Wd, Hd, fov=180.0):
    fov = fov * np.pi / 180.0

    # cartesian coordinates of the projected (square) image
    ys, xs = np.indices((Hs, Ws), np.float32)
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0

    # spherical coordinates
    theta, phi = get_theta_phi_2(x_proj, y_proj, Ws, Hs, fov)

    # polar coordinates (of the fisheye image)
    p = Hd * phi / fov

    # cartesian coordinates of the fisheye image
    y_fish = p * np.sin(theta)
    x_fish = p * np.cos(theta)

    ymap = Hd / 2.0 - y_fish
    xmap = Wd / 2.0 + x_fish
    return xmap, ymap



def get_overlap_regions(im_left,im_right,fov=193,lr_persent = 0.5):
	hl,wl = im_left.shape[:2] 
	overlaped_width = int((fov - 180)/fov * wl)
	l_overlaped_width = int((fov - 180)/fov * wl * lr_persent)
	r_overlaped_width = overlaped_width - l_overlaped_width
	
	im_left_left = im_left[:,:r_overlaped_width,:]
	im_left_right = im_left[:,wl - l_overlaped_width : wl,:]
	
	im_right_left = im_right[:,:l_overlaped_width,:]
	im_right_right = im_right[:,wl - r_overlaped_width : wl,:]
	

	
	return im_left_left,im_left_right,im_right_left,im_right_right
	
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

	
def draw_optical_pts(src_org,dst_org):
	src_track = cv2.cvtColor(src_org, cv2.COLOR_BGR2GRAY)
	dst_track = cv2.cvtColor(dst_org, cv2.COLOR_BGR2GRAY)
	
	# src_clone = src_org
	# dst_clone = dst_org
	src_clone = np.copy(src_org)
	dst_clone = np.copy(dst_org)
	
	
	p_src = cv2.goodFeaturesToTrack(src_track, mask = None, **feature_params)
	
	
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
	
	
	
def shift_ndarray_pts(arr,offset,axis  = 0):
	matrix = np.asmatrix(arr)
	matrix[:,axis] = matrix[:,axis] + offset
	new_arr = np.asarray(matrix).reshape(-1,1,2)
	
	return new_arr

def right_image_to_middle(im_right,ol_l,ol_r):
	hr,wr = im_right.shape[:2]
	result = np.zeros((hr, wr*2 - (ol_l+ol_r),3), np.uint8)
	wr_half = int(wr/2)
	result[:,wr_half - ol_l:wr_half+wr - ol_l,:] = im_right
	return result
	
	
def stitch_org(old_img,good_old,new_img,good_new,overlap_width_l,overlap_width_r):
	if len(good_new)>MIN_MATCH_COUNT:
		src_pts = np.float32([ good_new]).reshape(-1,1,2)
		dst_pts = np.float32([ good_old]).reshape(-1,1,2)


		M, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask1.ravel().tolist()

		height,width = new_img.shape[:2]
		
		#image that focus on the left part
		warp_img = cv2.warpPerspective(new_img, M, (width*2, height))
		
		overlap_width = overlap_width_l + overlap_width_r
		

		warp_img = np.roll(warp_img,width - overlap_width,axis=1)
		warp_img[0:height,0:width,:] = old_img
		# warp_img[0:height,width-overlap_width:width,:] = overlap_lr
		
		cv2.imshow('temlate_points1',warp_img)
		cv2.imwrite('video2/stitch_result/temlate_points_warp.png',warp_img)
		cv2.waitKey(0)	

	else:
		print ("draw stitched Not enough matches are found - %d/%d" )
		matchesMask = None
	


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
		mask = xy_blend.imgLabeling2(im_lr,im_rl,im_ll,im_rr,x_offset_l,x_offset_r)
		plt.imshow(mask),plt.show()
		labeled = warp_img2 * mask  + warp_img1 * (1 - mask)
		
		blended = xy_blend.multi_band_blending(warp_img2, warp_img1, mask, 2.0)
		cv2.imshow('p', blended.astype(np.uint8))
		cv2.waitKey(0)
		cv2.imwrite('video2/stitch_result/stitch_result_l_'+str(overlap_width_l)+'_r'+str(overlap_width_r)+'_multi_blend.png',blended)
		
		
		cv2.imshow('labeled',labeled.astype(np.uint8))
		cv2.waitKey(0)
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

		
		# warp_img[0:height,0:w_half,:] = old_img[:,w_half:width,:]
		# warp_img[0:height,width*2 - w_half - (overlap_width_l+overlap_width_r) :width*2 - (overlap_width_l+overlap_width_r),:] = old_img[:,:w_half,:]
	
		warp_img[:,0:w_half - overlap_width_l,:] = old_img[:,w_half:width-overlap_width_l,:]
		warp_img[:,width*2 - w_half - (overlap_width_l) :width*2 - (overlap_width_l+overlap_width_r),:] = old_img[:,overlap_width_r:w_half,:]
		
		#overlap part		
		warp_img[:,w_half-overlap_width_l:w_half - 40,:] = old_img[:,width-overlap_width_l:width-40]
		warp_img[:,int(1.5*width) - overlap_width_l - 40: int(1.5*width) - overlap_width_l,:] = old_img[:,overlap_width_r-40:overlap_width_r,:]
		
		return warp_img
		
	

	else:
		print ("draw stitched Not enough matches are found - %d/%d" )
