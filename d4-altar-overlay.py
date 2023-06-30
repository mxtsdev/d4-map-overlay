import cv2
import json
import math
import time
import numpy as np
from mss import mss
import pygame as pg
import win32gui
import win32con
import win32api
import argparse

class MapCoords():
	def __init__(self):
		self.MAX_LATITUDE = 85.0511287798
		self.R = 6378137
		self._a = 2.495320233665337e-8
		self._b = 0.5
		self._c = -2.495320233665337e-8
		self._d = 0.5

	def latLngToPoint(self, pt, zoom = 1):
		x, y = self.project(pt)
		n = self.scale(zoom)
		return self.transform(x, y, n)
	
	def scale(self, n):
		return 256 * math.pow(2, n)
	
	def project(self, pt):
		lat = pt[0]
		lng = pt[1]
		i = math.pi / 180
		e = self.MAX_LATITUDE
		n = max(min(e, lat), -e)
		o = math.sin(n * i)
		return self.R * lng * i, self.R * math.log((1 + o) / (1 - o)) / 2
	
	def transform(self, x, y, n):
		return n * (self._a * x + self._b), n * (self._c * y + self._d)

class MapSIFTMatcher():
	def __init__(self, map_image):
		self.sift = cv2.SIFT_create()
		self.map_image = map_image
		self.kp_map, self.des_map = self.sift.detectAndCompute(map_image, None)

		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def query(self, query_image, markers, MIN_MATCH_COUNT = 10, debug = False):
		img = None
		height, width = None, None
		out_markers = []

		kp_query, des_query = self.sift.detectAndCompute(query_image, None)
		matches = self.flann.knnMatch(des_query, self.des_map, k=2)

		good = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				good.append(m) 

		if len(good) > MIN_MATCH_COUNT:
			h, w, d = query_image.shape

			src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
			dst_pts = np.float32([ self.kp_map[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			self.last_homography_matrix = M

			matchesMask = mask.ravel().tolist()

			try:
				transform_points = cv2.perspectiveTransform(np.float32([[0, 0], [0, h-1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2), M)
			except:
				matchesMask = None
				if debug == True:
					temp = self.map_image.copy()
			else:
				topleft,bottomleft,bottomright,topright = [np.int32(x[0]) for x in transform_points]
				width = bottomright[0] - topleft[0]
				height = bottomright[1] - topleft[1]

				if debug == True:
					temp = cv2.polylines(self.map_image.copy(), [np.int32(transform_points)], True, (255, 0, 0), 3, cv2.LINE_AA)

		else:
			matchesMask = None
			if debug == True:
				print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
				temp = self.map_image.copy()

		if debug == True:
			draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
			query_image_temp = query_image.copy()
		
		for m in markers:
			x = int(round(m[0], 0))
			y = int(round(m[1], 0))
			if debug == True:
				cv2.circle(temp, (x, y), 10, (255,255,255), -1)

			if width != None:
				if x >= topleft[0] and x <= topright[0] and y >= topleft[1] and y <= bottomleft[1]:
					x -= topleft[0]
					y -= topleft[1]
					x = int(round(x * (w / width), 0))
					y = int(round(y * (h / height), 0))
					if debug == True:
						cv2.circle(query_image_temp, (x, y), 10, (255, 255, 255), -1)
					
					out_markers.append([x, y])

		if debug == True:
			img = cv2.drawMatches(query_image_temp, kp_query, temp, self.kp_map, good, None, **draw_params)
			img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

		return img, out_markers, len(good)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-large-map', default=False, type=lambda x: x.lower() not in ['false', 'no', '0', 'None'], help='use large map')
	parser.add_argument('--debug', default=False, type=lambda x: x.lower() not in ['false', 'no', '0', 'None'], help='toggle debug mode')
	opt = parser.parse_args()

	if opt.use_large_map == True:
		zoom = 6
		map_image_path = 'map_images/map_5.jpg'
	else:
		zoom = 5
		map_image_path = 'map_images/map_5_small.jpg'

	map_scale = 0.5
	map_image = cv2.imread(map_image_path)
	map_image = cv2.resize(map_image, (0,0), fx = map_scale, fy = map_scale)

	mc = MapCoords()
	markers = []
	with open('data.json', 'r') as f:
		data = json.load(f)
		jmarkers = data["markers"]
		for m in jmarkers:
			if m["type"] == "Altar of Lilith":
				x, y = mc.latLngToPoint(m["coords"], zoom) # 5 = small map, 6 = large map ::: tiled map (saved from web) zoom level + 1
				x /= (1 / map_scale)
				y /= (1 / map_scale)
				markers.append([x, y])

	stm = MapSIFTMatcher(map_image)
	pg.init()
	info = pg.display.Info()
	screen = pg.display.set_mode((info.current_w, info.current_h), pg.FULLSCREEN)
	screen_rect = screen.get_rect()
	FPS = 1
	clock = pg.time.Clock()

	transparency_color = (255, 0, 128)
	hwnd = pg.display.get_wm_info()["window"]
	styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
	styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOOLWINDOW
	win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
	win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*transparency_color), 0, win32con.LWA_COLORKEY)
	win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0,0,0,0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

	screen.fill((255, 0, 128))
	pg.display.flip()

	sct = mss()
	w, h = info.current_w, info.current_h
	monitor = {'top': 0, 'left': 0, 'width': w, 'height': h}
	while 1:
		start_ss = time.time()
		screenshot = np.array(sct.grab(monitor))
		screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
		end_ss = time.time()

		start_query = time.time()
		debug_img, out_markers, num_matches = stm.query(screenshot, markers, MIN_MATCH_COUNT =  40, debug = opt.debug)
		end_query = time.time()
		if opt.debug == True:
			print(f"Query: {end_query-start_query}s, Grab: {end_ss-start_ss}s, Num matches: {num_matches}")

		screen.fill((255, 0, 128))
		pg.draw.circle(screen, '#00ff00', (10, 10), 5)
		for m in out_markers:
			pg.draw.circle(screen, '#ff00ff', (m[0] + monitor["left"], m[1] + monitor["top"]), 3)

		pg.display.flip()

		clock.tick(FPS)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	pg.quit()
	cv2.destroyAllWindows()