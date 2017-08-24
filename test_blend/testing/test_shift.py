import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_edge(bin_mask,x_offset=10):
	h_b,w_b = bin_mask.shape[:2]
	edge_l = np.zeros((h_b,w_b,3))
	edge_r = np.zeros((h_b,w_b,3))
	edge_mask_l = np.zeros((h_b,w_b,3))
	edge_mask_r = np.zeros((h_b,w_b,3))
	for i in range(h_b):
		row,col = np.where(bin_mask[i,:,0].reshape(1,w_b) == 0)
		#print (str(i) + ":"+ str(blank_arr[0]) + ":" + str(blank_arr[len(blank_arr) - 1]))
	
		edge_l[i][col[0] - 1][:] = 1
		edge_r[i][col[len(col) - 1] - 1][:] = 1
		edge_mask_l[i][col[0] :col[0] + x_offset][:] = 1
		edge_mask_r[i][col[len(col)-1] + x_offset + 1:col[len(col)-1] + x_offset*2 + 1][:] = 1
		
	return edge_l,edge_r,edge_mask_l,edge_mask_r
	
def reshape_img(img_org1,img_org2,graph_cut_mask,x_offset):
	#get the edge of the image for extending
	m_h,m_w = img_org1.shape[:2]
	graph_cut_mask_r = np.roll(graph_cut_mask,x_offset,axis=1).astype(bool)
	graph_cut_mask_r_2 = np.roll(graph_cut_mask,x_offset*2,axis=1).astype(bool)

	# binary_edge_l = (graph_cut_mask_r + graph_cut_mask).astype(bool) - graph_cut_mask.astype(bool)
	# binary_edge_r = (graph_cut_mask_r_2 + graph_cut_mask_r).astype(bool) - graph_cut_mask_r_2
	
	
	edge_l_mask,edge_r_mask,binary_edge_l,binary_edge_r = get_edge(graph_cut_mask,x_offset)

	color_edge_line_2 = img_org2[np.where(edge_l_mask != 0)].reshape((1280,3))
	color_edge_region_2_l = np.zeros((1280,2560,3))
	color_edge_region_2_l[:,:,0] = (binary_edge_l[:,:,0].transpose() * color_edge_line_2[:,0]).transpose()
	color_edge_region_2_l[:,:,1] = (binary_edge_l[:,:,1].transpose() * color_edge_line_2[:,1]).transpose()
	color_edge_region_2_l[:,:,2] = (binary_edge_l[:,:,2].transpose() * color_edge_line_2[:,2]).transpose()
	
	coror_edge_rine_2 = img_org2[np.where(edge_r_mask != 0)].reshape((1280,3))
	coror_edge_region_2_r = np.zeros((1280,2560,3))
	coror_edge_region_2_r[:,:,0] = (binary_edge_r[:,:,0].transpose() * coror_edge_rine_2[:,0]).transpose()
	coror_edge_region_2_r[:,:,1] = (binary_edge_r[:,:,1].transpose() * coror_edge_rine_2[:,1]).transpose()
	coror_edge_region_2_r[:,:,2] = (binary_edge_r[:,:,2].transpose() * coror_edge_rine_2[:,2]).transpose()
	
	color_edge_line_1 = img_org1[np.where(edge_l_mask != 0)].reshape((1280,3))
	color_edge_region_1_l = np.zeros((1280,2560,3))
	color_edge_region_1_l[:,:,0] = (binary_edge_l[:,:,0].transpose() * color_edge_line_1[:,0]).transpose()
	color_edge_region_1_l[:,:,1] = (binary_edge_l[:,:,1].transpose() * color_edge_line_1[:,1]).transpose()
	color_edge_region_1_l[:,:,2] = (binary_edge_l[:,:,2].transpose() * color_edge_line_1[:,2]).transpose()
	
	coror_edge_rine_1 = img_org1[np.where(edge_r_mask != 0)].reshape((1280,3))
	coror_edge_region_1_r = np.zeros((1280,2560,3))
	coror_edge_region_1_r[:,:,0] = (binary_edge_r[:,:,0].transpose() * coror_edge_rine_1[:,0]).transpose()
	coror_edge_region_1_r[:,:,1] = (binary_edge_r[:,:,1].transpose() * coror_edge_rine_1[:,1]).transpose()
	coror_edge_region_1_r[:,:,2] = (binary_edge_r[:,:,2].transpose() * coror_edge_rine_1[:,2]).transpose()
	
	new_img1 = np.roll((1-graph_cut_mask)*img_org1,x_offset,axis=1) + color_edge_region_1_l + coror_edge_region_1_r
	
	graph_cut_mask_l = graph_cut_mask.copy()
	graph_cut_mask_r = graph_cut_mask.copy()
	graph_cut_mask_l[:,int(m_w/2):,:] = 0
	graph_cut_mask_r[:,:int(m_w/2),:] = 0
	graph_cut_mask_r = np.roll(graph_cut_mask_r,x_offset*2,axis=1)
	new_img2 = img_org2*graph_cut_mask_l + color_edge_region_2_l + coror_edge_region_2_r + np.roll(img_org2,x_offset*2,axis=1)*graph_cut_mask_r
	
	#to be implement
	print(len(np.where(binary_edge_l!=0)[0]))
	cv2.imshow('binary_edge_l',(binary_edge_l*255).astype(np.uint8))
	a = np.arange(0.0,1.0,1.0/x_offset)
	b = np.repeat(a[np.newaxis,:], m_h, axis=0)
	c = np.repeat(b[:, :, np.newaxis], 3, axis=2)
	
	blend_mask_l = binary_edge_l.copy().astype(np.float32)
	blend_mask_r = binary_edge_r.copy().astype(np.float32)
	print(len((1-c).ravel()))
	blend_mask_l[np.where(binary_edge_l == 1)] = (1-c).ravel()
	blend_mask_r[np.where(binary_edge_r == 1)] = c.ravel()
	
	new_mask = blend_mask_l + blend_mask_r + graph_cut_mask_l + graph_cut_mask_r
	#new_mask = graph_cut_mask_l + graph_cut_mask_r
	return new_img1,new_img2,new_mask
	
	
		

#retrie the images
if __name__ == "__main__":
	
	warp_img2 = cv2.imread('video2/warp_img2.png')
	warp_img1 = cv2.imread('video2/warp_img1.png')
	warp_img1_no_hole = cv2.imread('video2/warp_img1_no_hole.png')
	graph_cut_mask = cv2.imread('video2/stitch_result/graphcut_mask_multi.png')
	
	
	x_offset = 3
	
	
	new_img1,new_img2,new_mask = reshape_img(warp_img1,warp_img2,graph_cut_mask,x_offset)
	#try to adjust the intensity
	new_img1 = new_img1 * 1.008
	
	new_blend = new_mask * new_img2 + (1-new_mask)*new_img1
	cv2.imshow('new_blend',new_blend.astype(np.uint8))
	cv2.imwrite('video2/new_img1.png',new_img1)
	cv2.imwrite('video2/new_img2.png',new_img2)
	cv2.imwrite('video2/new_mask.png',(new_mask*255).astype(np.uint8))
	cv2.imwrite('video2/new_blend.png',new_blend)
	
	
	test = np.arange(12).reshape((3,4))
	
	
	test_copy = test.copy()
	test_copy[2:,:] = 0
	print(test_copy)
	print(test)
	
	
	test_where = np.eye(6) + np.roll(np.eye(6),1,axis=1)
	print(len(np.where(test_where == 1)[0]))
	test_where[np.where(test_where == 1)] = np.arange(1,13)
	print(test_where)
	
	a = np.arange(0.0,1.0,1.0/x_offset)
	b = np.repeat(a[np.newaxis,:], 1280, axis=0)
	c = np.repeat(b[:, :, np.newaxis], 3, axis=2)
	
	
	# blend_region = np.zeros((1280,80,3))
	# print(np.arange(0.0,1.0,1.0/x_offset))
	# blend_region = blend_region * np.arange(0.0,1.0,1.0/x_offset)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()