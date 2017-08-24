#!/usr/bin/python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random



def buildmap_pgm(pgm_addr):
    pgm = open(pgm_addr)
    lines = pgm.readlines()
    Wd = int(lines[2].split(' ')[0])
    Hd = int(lines[2].split(' ')[1])
    result_map = np.zeros((Hd, Wd), np.float32)
    for y in range(4, 4 + Hd):
        locs = lines[y].split(' ')
        for x in range(Wd):
            result_map.itemset((y - 4, x), int(locs[x]))
    return result_map
	


def buildmap(Ws, Hs, Wd, Hd, hfovd=160.0, vfovd=160.0):
    # Build the fisheye mapping
    map_x = np.zeros((Hd, Wd), np.float32)
    map_y = np.zeros((Hd, Wd), np.float32)
    vfov = (vfovd / 180.0) * np.pi
    hfov = (hfovd / 180.0) * np.pi
    vstart = ((180.0 - vfovd) / 180.00) * np.pi / 2.0
    hstart = ((180.0 - hfovd) / 180.00) * np.pi / 2.0
    count = 0
    # need to scale to changed range from our
    # smaller cirlce traced by the fov
    xmax = np.cos(hstart)
    xmin = np.cos(hstart + hfov)
    xscale = xmax - xmin
    xoff = xscale / 2.0
    zmax = np.cos(vstart)
    zmin = np.cos(vfov + vstart)
    zscale = zmax - zmin
    zoff = zscale / 2.0
    # Fill in the map, this is slow but
    # we could probably speed it up
    # since we only calc it once, whatever
    for y in range(0, int(Hd)):
        for x in range(0, int(Wd)):
            count = count + 1
            phi = vstart + (vfov * (float(y) / float(Hd)))
            theta = hstart + (hfov * (float(x) / float(Wd)))
            xp = (np.sin(phi) * np.cos(theta) + xoff) /  xscale
            zp = (np.cos(phi) + zoff) / zscale
            xS = Ws - (xp * Ws)
            yS = Hs - (zp * Hs)
            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))

    return map_x, map_y
	
def testwarp(Ws,Hs,Wd,Hd):
	map_x = np.zeros((Hd, Wd), np.float32)
	map_y = np.zeros((Hd, Wd), np.float32)
	for y in range(0,int(Hd)):
		for x in range(0,int(Wd)):
			xs = 0.5*x+100
			ys = random.randint(1, 100)
			map_x.itemset((int(ys),int(xs)),int(x))
			map_y.itemset((int(ys),int(xs)),int(y))
	return map_x,map_y

def polar_to_sphere(Ws,Hs,Wd,Hd,fov):
	map_x = np.zeros((Hd, Wd), np.float32)
	map_y = np.zeros((Hd, Wd), np.float32)
	
	radius = Hs/2
	for y in range(0,int(Hs)):
		for x in range(0,int(Ws)):
			#convert to unit cartesian coordinate
			xc = (x - Ws/2) / radius
			yc = -(y - Hs/2) / radius
			
			#convert to polar coordinate
			p_r = math.sqrt(math.pow(xc,2)+math.pow(yc,2)) 
			p_theta = np.arctan2(yc,xc)
			
			# map_x.itemset((int(p_r),int(p_theta * 50)),int(x))
			# map_y.itemset((int(p_r),int(p_theta * 50)),int(y))
			
			if(p_r > 1):
				continue
			
			#polar to unit 3d sphere
			z_sph = math.sqrt(1-math.pow(p_r,2))
			phi_sph = np.arccos(z_sph)
			
			#project to
			p = phi_sph / fov * Hs
			
			x_proj = p*np.cos(p_theta) + 0.5 * Ws
			y_proj = p*np.sin(p_theta) + 0.5 * Hs
			
			if((int(x_proj) >= Ws) or (int(y_proj) >= Hs)):
				print("x:"+str(x)+"y:"+str(y)+"x_proj:"+str(x_proj)+"y_proj"+str(y_proj))
				continue
			
			map_x.itemset((int(y_proj),int(x_proj)),x)
			map_y.itemset((int(y_proj),int(x_proj)),y)
			
			# map_x.itemset((y,x),int(x_proj))
			# map_y.itemset((y,x),int(y_proj))

			
			


	return map_x,map_y
	
#construct the 3d sphere according the method shown in the paper
def get_forwardmap_origin(Ws,Hs,Wd,Hd,fov):
	map_x = np.zeros((Hd, Wd), np.float32)
	map_y = np.zeros((Hd, Wd), np.float32)
	
	radius = Hs/2
	for y in range(0,int(Hs)):
		for x in range(0,int(Ws)):
			#convert to unit cartesian coordinate
			xc = (x - Ws/2)
			yc = -(y - Hs/2)
			# xc  = x
			# yc = y
			
			theta_s = fov*xc/Ws - 0.5
			phi_s = fov*yc/Hs - 0.5
			
			xp = np.cos(phi_s)*np.sin(theta_s)
			yp = np.cos(phi_s)*np.cos(theta_s)
			zp = np.sin(phi_s)
			
			theta = np.arctan2(zp,xp)
			phi = np.arctan2(math.sqrt(math.pow(xp,2)+math.pow(zp,2)),yp)
			p = Hs/fov*phi
			
			#convert to projection space
			x_proj = 0.5*Ws + p*np.cos(theta)
			y_proj = 0.5*Hs + p*np.sin(theta)
			
			if((x_proj >= Wd ) or (y_proj >= Hd)):
				print ("x:"+str(x_proj)+" y:"+str(y_proj))
				continue
				
			map_x.itemset((int(y_proj),int(x_proj)),x)
			map_y.itemset((int(y_proj),int(x_proj)),y)
			
	return map_x,map_y



#longitude-latitude reverse or forward map correction method
def latitudeCorrection(Hs,Ws,cameraFieldAngle):
	map_x = np.zeros((Hs, Ws), np.float32)
	map_y = np.zeros((Hs, Ws), np.float32)
	
	dx = cameraFieldAngle/Ws
	dy = dx
	longitude_offset = (np.pi - cameraFieldAngle)/2
	latitude_offset = (np.pi - cameraFieldAngle)/2 
	
	radius_of_src = math.sqrt(math.pow(Hs/2,2)+math.pow(Ws/2,2))
	R = radius_of_src / np.sin(cameraFieldAngle/2)
	center = np.array([Ws/2,Hs/2])
	#Convert to cartiesian cooradinate in unity circle
	for j in range(0,Hs):
		for i in range(0,Ws):
			u = i
			v = j
			#convert image to catrisan in unit circle
			x_cart = (u - center[0])/radius_of_src
			y_cart = -(v - center[1])/radius_of_src
			
			#print("center:"+str(center[0])+" "+str(center[1])+"x_cart:"+str(x_cart)+"y_cart:"+str(y_cart)+'U:'+str(u)+'v:'+str(v))
			
			#convert to polar
			theta = np.arctan2(y_cart,x_cart)
			p = math.sqrt(math.pow(x_cart,2)+math.pow(y_cart,2))
			
			#convert to sphere surface parameter coordinate
			theta_sphere = np.arcsin(p)
			phi_sphere = theta
			
			
			#convert to sphere surface 3D coordinate
			x = np.sin(theta_sphere)*np.cos(phi_sphere)
			y = np.sin(theta_sphere)*np.sin(phi_sphere)
			z = np.cos(theta_sphere)
			
			#convert to latitude cooradinate
			latitude = np.arccos(y)
			longitude = np.arctan2(z,(-x))
			
			u_latitude = (longitude - longitude_offset)/dx
			v_latitude = (latitude - latitude_offset)/dy
			# print("u_latitude:"+str(u_latitude)+"v_latitude:"+str(v_latitude))
			
			if(u_latitude < 0 ) or ( u_latitude >= Hs ) or ( v_latitude < 0 ) or ( v_latitude >= Ws):
				continue
			if(np.isnan(u_latitude)) or (np.isnan(v_latitude)) :
				print ("nan j :"+str(j)+" i:"+str(i))
				continue
				
			map_x.itemset((j, i), int(v_latitude))
			map_y.itemset((j, i), int(u_latitude))
			# map_x.itemset((int(v_latitude),int(u_latitude)),x)
			# map_y.itemset((int(v_latitude),int(u_latitude)),y)
			
			
			
	return map_x, map_y

	
	#modified version  confused point	
def get_forwardmap_modified(Ws,Hs,Wd,Hd,fov):
	map_x = np.zeros((Hd, Wd), np.float32)
	map_y = np.zeros((Hd, Wd), np.float32)
	
	radius = Hs/2
	for y in range(0,int(Hs)):
		for x in range(0,int(Ws)):
			#convert to unit cartesian coordinate
			xc = (x - Ws/2)
			yc = -(y - Hs/2)

			
			theta_s = fov*xc/Ws - (0.5/180*np.pi)
			phi_s = fov*yc/Hs - (0.5/180*np.pi)
			
			xp = np.cos(phi_s)*np.sin(theta_s)
			yp = np.cos(phi_s)*np.cos(theta_s)
			zp = np.sin(phi_s)
			
			# map_x.itemset((int(yp*800 - 1),int(zp*800 - 1)),x)
			# map_y.itemset((int(yp*800 - 1),int(zp*800 - 1)),y)
			# map_x.itemset((y,x),int(zp*600 - Ws/2))
			# map_y.itemset((y,x),int(Hs/2 - yp*600))
			
			# theta = np.arctan2(zp,xp)
			# phi = np.arctan2(math.sqrt(math.pow(xp,2)+math.pow(zp,2)),yp)
			
			
			#project to yz plane
			# phi = np.arctan2(math.sqrt(math.pow(yp,2)+math.pow(zp,2)),xp)
			# theta = np.arctan2(zp,yp)
			
			#project to xz plane
			theta = np.arctan2(zp,xp)
			phi = np.arctan2(math.sqrt(math.pow(xp,2)+math.pow(zp,2)),yp)
			
			p = Hs/fov*phi
			x_proj = p*np.cos(theta) + 0.5*Ws
			y_proj = 0.5*Hs - p*np.sin(theta)
			
			# if((x_proj >= Wd ) or (y_proj >= Hd)):
				# print ("x:"+str(x)+"y:"+str(y)+"x_proj:"+str(x_proj)+" y_proj:"+str(y_proj))
				# continue
				
			#check forward map	
			map_x.itemset((y,x),int(x_proj))
			map_y.itemset((y,x),int(y_proj))
				
			# map_x.itemset((int(y_proj),int(x_proj)),x)
			# map_y.itemset((int(y_proj),int(x_proj)),y)
			
	return map_x,map_y

def nothing(x):
    pass 

if __name__ == "__main__":
	img = cv2.imread('fisheye_grid_2.jpg', cv2.IMREAD_COLOR)
	height, width = img.shape[:2]
	# xmap, ymap = buildmap(Ws=width, Hs=height, Wd=width,
                          # Hd=height, hfovd=180, vfovd=180)
						  
						  
	#to do test inverse 					  
	#xmap,ymap = latitudeCorrection(Ws=width, Hs=height, Wd=width,Hd=height)
	#xmap,ymap = latitudeCorrection(height,width,160/180*np.pi)
	test_name = "xzplane_proj_forward"
	xmap,ymap = get_forwardmap_modified(width,height,width,height,170/180*np.pi)
	np.savetxt('xmap.txt',xmap)
	np.savetxt('ymap.txt',ymap)
	result = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
	cv2.imwrite('warp_results/'+test_name+'_angular_correction.jpg',result)
	cv2.imwrite('xymaps/'+test_name+'_xmap.png',xmap)
	cv2.imwrite('xymaps/'+test_name+'_ymap.png',ymap)
	plt.imshow(xmap, cmap='gray', interpolation='nearest')
	plt.imshow(ymap, cmap='gray', interpolation='nearest')
	plt.show()
	
			


