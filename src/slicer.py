import os
import cv2
import numpy as np

from shapely import geometry


class Slicer():

    def __init__( self, width=544, height=544, overlap=0.1 ):

        """
        constructor - setup default sub-image dimensions and overlap
        """

        # slicing variables
        self._overlap = overlap
        self._zero_frac_thresh = 0.2

        self._width = width
        self._height = height

        return


    def process( self, img_pathname, out_path, aoi=None, pad=0 ):

        """
        slice large image into smaller chunks and save to output path
        optional pixel-based area of interest and border padding
        """

        # read image
        image = cv2.imread( img_pathname, 1 )  # color
        slices = []

        # get size of image and slice size
        im_h, im_w = image.shape[:2]
        win_size = self._height * self._width

        # if slice sizes are large than image, pad the edges
        if self._height > im_h:
            pad = self._height - im_h

        if self._width > im_w:
            pad = max(pad, self._width - im_w)

        # pad the edge of the image with black pixels
        if pad > 0:
            border_color = (0, 0, 0)
            image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, value=border_color)

        # initialise stride - maintain overlap between slices
        dx = int((1. - self._overlap) * self._width )
        dy = int((1. - self._overlap) * self._height )

        # 10% overlap between image slices
        for y in range(0, im_h, dy):
            for x in range(0, im_w, dx):

                # get new sub-image origin - careful not to past image edge
                y0 = ( im_h - self._height ) if y + self._height > im_h else y
                x0 = ( im_w - self._width ) if x + self._width > im_w else x

                if aoi is None or self.isIntersection( aoi, [ y0, x0, y0 + self._height, x0 + self._width ] ): 

                    # get window into image
                    window_c = image[y0:y0 + self._height, x0:x0 + self._width]
                    win_h, win_w = window_c.shape[:2]

                    # create output pathname
                    filename = 'slice_{}_{}_{}_{}_{}_{}'.format( os.path.splitext( os.path.basename( img_pathname ) )[0], y0, x0, win_h, win_w, pad )
                    out_pathname = os.path.join( out_path, filename + '.jpg')

                    if not os.path.exists( out_pathname ):

                        # get black and white sub-image
                        window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

                        # find threshold of sub-image that's not black
                        ret, thresh = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
                        non_zero_counts = cv2.countNonZero(thresh)

                        zero_counts = win_size - non_zero_counts
                        zero_frac = float(zero_counts) / win_size

                        # skip if sub-image is mostly empty
                        if zero_frac >= self._zero_frac_thresh:
                            continue

                        # create output folder in readiness
                        if not os.path.exists( out_path ):
                            os.makedirs( out_path )

                        # save sub-image as slice
                        cv2.imwrite(out_pathname, window_c)

                    # save sub-image as slice
                    slices.append( {    'pathname' : out_pathname, 
                                        'y0' : y0, 
                                        'x0' : x0, 
                                        'win_h' : win_h,
                                        'win_w' : win_w,
                                        'height' : self._height,
                                        'width' : self._width,
                                        'pad' : pad } )

        return slices


    def isIntersection( self, r1, r2 ):

        """
        boolean check if rectangles overlap in cartesian space
        """

        # calculate intersection between aoi and slice window rectangles
        p1 = geometry.Polygon([(r1[0],r1[1]), (r1[1],r1[1]),(r1[2],r1[3]),(r1[2],r1[1])])
        p2 = geometry.Polygon([(r2[0],r2[1]), (r2[1],r2[1]),(r2[2],r2[3]),(r2[2],r2[1])])

        return(p1.intersects(p2))

