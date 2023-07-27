import cv2
import numpy as np

class Helpers:
	def __init__(self):
		pass

	def resize(self, width=None, height=None, inter=cv2.INTER_AREA):
		dim = None
		(h, w) = self.shape[:2]
		if width is None and height is None:
			return self
		if width is None:
		    r = height / float(h)
		    dim = (int(w * r), height)
		else:
		    r = width / float(w)
		    dim = (width, int(h * r))
		return cv2.resize(self, dim, interpolation=inter)

	def grab_contours(self):
		if len(self) == 2:
			self = self[0]
		elif len(self) == 3:
			self = self[1]
		else:
			raise Exception('The length of the contour must be 2 or 3.')
		return self


	def orders(self):
		rect = np.zeros((4, 2), dtype = "float32")
		s = self.sum(axis = 1)

		rect[0] = self[np.argmin(s)]
		rect[2] = self[np.argmax(s)]

		diff = np.diff(self, axis = 1)
		rect[1] = self[np.argmin(diff)]
		rect[3] = self[np.argmax(diff)]

		return rect

	def transform(self, pts):
		rect = Helpers.orders(pts)
		(tl, tr, br, bl) = rect

		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))

		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

		M = cv2.getPerspectiveTransform(rect, dst)
		return cv2.warpPerspective(self, M, (maxWidth, maxHeight))