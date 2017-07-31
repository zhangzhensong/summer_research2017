import cv2
import numpy as np
from matplotlib import pyplot as plt
import xw_utils
import opt_flow

def get_overlap_regions3(im_left,im_right,overlap_width_l,overlap_width_r,y_offset):
	hl,wl = im_left.shape[:2]

	im_left_left = im_left[y_offset:hl-y_offset,:overlap_width_r,:]
	im_left_right = im_left[y_offset:hl-y_offset,wl - overlap_width_l : wl,:]
	
	im_right_left = im_right[y_offset:hl-y_offset,:overlap_width_l,:]
	im_right_right = im_right[y_offset:hl-y_offset,wl - overlap_width_r : wl,:]
		
	return im_left_left,im_left_right,im_right_left,im_right_right

def draw_pts_and_compare(src_org,dst_org,src_pts,dst_pts):
	src_clone = np.copy(src_org)
	dst_clone = np.copy(dst_org)
	for i,(dst_pt,src_pt) in enumerate(zip(dst_pts,src_pts)):
		a,b = dst_pt.ravel()
		c,d = src_pt.ravel()
		cv2.circle(dst_clone,(int(a),int(b)),3,(0,0,255),-1)
		cv2.circle(src_clone,(int(c),int(d)),3,(0,0,255),-1)
	combined_im = np.concatenate((src_clone,dst_clone),axis=1)
	return combined_im

	
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    h = int(h)
    w = int(w)
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
	
def dense_compare(pre_org,new_org,min_dis,max_dis):
	prevgray = cv2.cvtColor(pre_org, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(new_org, cv2.COLOR_BGR2GRAY)
	
	#get the flow
	flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 42, 5, 5, 1.2, 1)
	im_flow = draw_flow(gray, flow,8)	
	#plt.imshow(im_flow),plt.show()
	
	offset_x = flow[:,:,0]
	offset_y = flow[:,:,1]
	dis = np.sqrt(np.power(offset_x,2)+np.power(offset_y,2))
	dis_avg = np.average(dis)
	mask = (dis > min_dis )*( dis < max_dis)
	plt.imshow(dis),plt.show()
	plt.imshow(mask),plt.show()
	pos_arr = np.where(mask)
	y,x = pos_arr
	good_pts_new = np.stack((x,y),axis = -1).reshape((len(y),1,2))
	

	offset_all = flow[mask].reshape((len(y),1,2))
	temp = np.copy(offset_all[:,:,0])
	offset_all[:,:,0] = offset_all[:,:,1]
	offset_all[:,:,1] = temp
	
	good_pts_org = good_pts_new + offset_all
	
	#test plot the points in the image
	combined_im = draw_pts_and_compare(pre_org,new_org,good_pts_org,good_pts_new)
	return combined_im,good_pts_org,good_pts_new

if __name__ == "__main__":
	unwarp_left = cv2.imread('video1/unwarp_left.png')
	unwarp_right = cv2.imread('video1/unwarp_right.png')
	
	h,w = unwarp_left.shape[:2]
	overlap_width_l = 62
	overlap_width_r = 50
	
	min_dis = 3
	max_dis = 16
	y_offset = 350
	
	
	im_ll,im_lr,im_rl,im_rr = get_overlap_regions3(unwarp_left,unwarp_right,overlap_width_l,overlap_width_r,y_offset)
	#test stitch the images with given pts
	combined_lr_rl,good_pts_lr,good_pts_rl = dense_compare(im_lr,im_rl,1,6)
	combined_rr_ll,good_pts_rr,good_pts_ll = dense_compare(im_rr,im_ll,3,16)
	cv2.imshow('combined_lr_rl',combined_lr_rl)
	
	cv2.imwrite('video1/test_result/compared_lr_rl.png',combined_lr_rl)
	cv2.imwrite('video1/test_result/compared_ll_rr.png',combined_rr_ll)
	
	
	#sh1ft the pts
	ll_rr_shift = int(w*1.5) - (overlap_width_l + overlap_width_r) 
	lr_rl_shift = int(w*0.5) - overlap_width_l 
	

	good_pts_lr = good_pts_lr + np.array((lr_rl_shift,y_offset))
	good_pts_rl = good_pts_rl + np.array((lr_rl_shift,y_offset))
	good_pts_rr = good_pts_rr + np.array((ll_rr_shift,y_offset))
	good_pts_ll = good_pts_ll + np.array((ll_rr_shift,y_offset))
	
	good_left = np.concatenate((good_pts_lr,good_pts_ll),axis = 0)
	good_right = np.concatenate((good_pts_rl,good_pts_rr),axis = 0)
	
	# good_left = good_pts_lr
	# good_right = good_pts_rl
	
	right_middle = xw_utils.right_image_to_middle(unwarp_right,overlap_width_l,overlap_width_r)
	cv2.imwrite('right_middle.png',right_middle)
	warp_img = xw_utils.stitch_to_middle(unwarp_left,good_left,right_middle,good_right,overlap_width_l,overlap_width_r)
	img_name = "good_lr_rl_only.png"
	cv2.imwrite('video1/stitch_result/'+img_name,warp_img)
	


	
	#find teh points with great shift
	
	
	cv2.waitKey(0)
	
	cv2.destroyAllWindows()