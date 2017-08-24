import cv2
import numpy as np
import utils
import networkx as nx
import sys
from matplotlib import pyplot as plt

INTMAX = sys.maxsize


def find_graph_cut(imlr, imrl, imrr, imll, maskSize,xoffsetL, xoffsetR):
	mask_l = get_single_mask(imlr,imrl)
	mask_r = get_single_mask(imrr,imll)
	mask_all = np.zeros((maskSize[1], maskSize[0], 3), np.float64)
	mask_all[:,:xoffsetL,:] = 1
	mask_all[:,xoffsetR+mask_r.shape[1]:,:] = 1
	mask_all[:,xoffsetL:xoffsetL+mask_l.shape[1],:] = mask_l
	mask_all[:,xoffsetR:xoffsetR+mask_r.shape[1],:] = 1 - mask_r
	return mask_all

def get_single_mask(i_l,i_r):
	h,w = i_l.shape[:2]
	
	abs_diff = np.absolute(i_l - i_r)
	color_diff = np.zeros((h,w),np.int32)
	color_diff = np.sqrt( np.power(np.int32(abs_diff[:,:,0]),2) + np.power(np.int32(abs_diff[:,:,1]),2) + np.power(np.int32(abs_diff[:,:,2]),2)  )
	
	grad_l = cv2.Canny(i_l,100,200)
	grad_r = cv2.Canny(i_r,100,200)
	grad_abs_sum = np.int32(np.absolute(grad_l)) + np.int32(np.absolute(grad_r))
	
	G1 = build_graph(i_l,i_r,color_diff,grad_abs_sum)
	
	cut_value, partition = nx.minimum_cut(G1, str(h*w), str(h*w + 1))
	reachable, non_reachable = partition
	
	
	mask = np.zeros((1,h*w))
	reachable_l = list(map(int,reachable))
	reachable_l.remove(h*w)
	mask[0][reachable_l] = 1
	mask = mask.reshape((h,w))
	mask_color = np.zeros((h,w,3))
	mask_color[:,:,0] = mask
	mask_color[:,:,1] = mask
	mask_color[:,:,2] = mask
	
	return mask_color

def build_graph(im_a,im_b,c_diff,grad_sum):
	h,w = im_a.shape[:2]
	G = nx.Graph()
	G.add_nodes_from(np.char.mod('%d', np.arange(h*w + 2)))
	
	idx_source = h*w
	idx_dst = h*w + 1
	
	
	indices = np.arange(h*w).reshape((h,w))
	indices_l, indices_r ,indices_top, indices_btm = get_shifted_matrix(indices) 
	color_diff_l, color_diff_r,color_diff_top,color_diff_btm = get_shifted_matrix(c_diff)
	grad_abs_sum_l,grad_abs_sum_r,grad_abs_sum_top,grad_abs_sum_btm = get_shifted_matrix(grad_sum)
	# cost_lr = (color_diff_l + color_diff_r) / (grad_abs_sum_l + grad_abs_sum_r+0.1)
	# cost_bt = (color_diff_top + color_diff_btm) / (grad_abs_sum_top + grad_abs_sum_btm + 0.1)
	
	cost_lr = (color_diff_l + color_diff_r) 
	cost_bt = (color_diff_top + color_diff_btm) 
	
	
	# print (indices_l)
	# print(indices_r)
	# print(cost_lr)
	dict_lr = [(str(x),str(y),{'capacity':z}) for x,y,z in zip(indices_l.ravel(),indices_r.ravel(),cost_lr.ravel()) ]
	dict_bt = [(str(x),str(y),{'capacity':z}) for x,y,z in zip(indices_top.ravel(),indices_btm.ravel(),cost_bt.ravel())] 
	G.add_edges_from(dict_lr)
	G.add_edges_from(dict_bt)
	
	#left most column infinity
	dict_left_most = [(str(x),str(idx_source),{'capacity':INTMAX}) for x in indices[:,0].ravel() ]
	G.add_edges_from(dict_left_most)
	# print (dict_left_most)
	
	#right most column infinity
	dict_right_most = [(str(x),str(idx_dst),{'capacity':INTMAX}) for x in  indices[:,w-1].ravel() ]
	G.add_edges_from(dict_right_most)
			
	return G

	
def get_shifted_matrix(m_org):
	h,w = m_org.shape[:2] 
	m_l = np.delete(m_org,w-1,1)
	m_r = np.delete(m_org,0,1)
	m_top = np.delete(m_org,h-1,0)
	m_btm = np.delete(m_org,0,0)
	
	return m_l,m_r,m_top,m_btm
