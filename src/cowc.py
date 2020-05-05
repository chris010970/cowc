import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

import dicttoxml
from xml.dom.minidom import parseString

from PIL import Image
from collections import OrderedDict
from slicer import Slicer


class Cowc:

    def __init__( self, size=(3,3) ):

        """
        placeholder
        """

        # list of train and test directories
        self._annotation_suffix = '_Annotated_Cars.png'

        # 15cm resolution
        self._GSD = 0.15
        self._size = ( int( round( ( size[ 0 ] / self._GSD ) / 2 ) ), int ( round ( ( size[ 1 ] / self._GSD ) / 2 ) ) )

        # xml conversion tweak
        self._custom_item_func = lambda x: 'object'

        # create image slicer
        self._slicer = Slicer()

        return


    def process( self, data_path, out_path ):

        """
        create images and annotations for train and validation
        """

        # for each subset
        for subset in [ 'train', 'test' ]:

            # locate all images in data path
            path = os.path.join( data_path, subset ) 
            files = glob.glob( os.path.join( os.path.join( path, '**' ), '*.png' ), recursive=True )    
            files = [ x for x in files if 'Annotated' not in x ]

            # slice up images
            for f in files:

                slices = self._slicer.process ( f, os.path.join( out_path, '{}/images'.format( subset ) ) )
                
                # check annotation image exists
                pathname = os.path.join( f.replace( '.png', self._annotation_suffix ) )
                if os.path.exists( pathname ):

                    # create PASCAL VOC schema for each image slice
                    annotation_image = cv2.imread( pathname )
                    for s in slices:
                        self.getAnnotation( s, annotation_image, os.path.join( out_path, '{}/annotations'.format( subset ) ) )

        return


    def getAnnotation( self, s, annotation_image, out_path, writeback=False, overwrite=True ):

        """
        create annotation xml files encoding bounding box locations
        """

        # create label pathname
        filename = os.path.splitext( os.path.basename( s[ 'pathname' ] ) )[ 0 ] + '.xml' 
        annotation_pathname = os.path.join( out_path, filename )

        if not os.path.exists( annotation_pathname ) or overwrite:

            # get bounding boxes for cars in aoi 
            results, label_locs = self.getBoundingBoxes( s, annotation_image )
            schema = self.getSchema( s, results )

            # create output dir if necessary
            if not os.path.exists( out_path ):
                os.makedirs( out_path )

            # write annotation to xml file
            with open( os.path.join( out_path, filename ), "w+" ) as outfile:

                # parse xml into string
                xml = dicttoxml.dicttoxml( schema, attr_type=False, item_func=self._custom_item_func, custom_root='annotation' ) \
                        .replace(b'<annotation>',b'<annotation verified="yes">') \
                        .replace(b'<items>',b'').replace(b'</items>',b'') \

                dom = parseString( xml )

                # write xml string to file
                outfile.write( dom.toprettyxml() )

            # plot writeback
            if writeback:
                self.drawBoundingBoxes( s[ 'pathname' ], results )

        return


    def getBoundingBoxes( self, s, annotation_image, heading='fixed' ):

        """
        extract bounding boxes around car locations from annotation image
        """

        # process each slice
        records = []

        # extract window from annotation image
        x0 = s[ 'x0' ]; y0 = s[ 'y0' ]
        window = annotation_image[ y0:y0 + s [ 'height' ], x0:x0 + s[ 'width' ] ]

        # find locations of non-zero pixels - add zero rotation column
        label_locs = np.where( window > 0)
        label_locs = np.transpose( np.vstack( [ label_locs[ 0 ], label_locs[ 1 ], np.zeros( len( label_locs[ 0 ] ) ) ]  ) )

        if label_locs.size > 0:

            # create bounding box for annotated car locations
            for loc in label_locs:    
                record = self.getBoundingBox( loc, window.shape )

                # ignore annotated objects close to image edge
                if record:
                    records.append( record )

        return records, label_locs


    def getBoundingBox( self, loc, dims ):

        """
        placeholder
        """
        
        # extrapolate bbox from centroid coords
        record = {}
        yc, xc, angle = loc

        # compute pts along vertical line rotated at mid point
        x0_r, y0_r = self.rotatePoint( xc, yc + self._size[ 1 ], xc, yc, math.radians( angle ) ) 
        x1_r, y1_r = self.rotatePoint( xc, yc - self._size[ 1 ], xc, yc, math.radians( angle ) ) 

        # compute corner pts orthogonal to rotated line end points
        corner = np.empty( (4, 2), float )

        corner[ 0 ] = self.rotatePoint( x0_r, y0_r + self._size[ 0 ], x0_r, y0_r, math.radians( angle + 90.0 ) )
        corner[ 1 ] = self.rotatePoint( x0_r, y0_r - self._size[ 0 ], x0_r, y0_r, math.radians( angle + 90.0 ) )

        corner[ 2 ] = self.rotatePoint( x1_r, y1_r + self._size[ 0 ], x1_r, y1_r, math.radians( angle + 90.0 ) )
        corner[ 3 ] = self.rotatePoint( x1_r, y1_r - self._size[ 0 ], x1_r, y1_r, math.radians( angle + 90.0 ) )

        # get min and max coordinates for bbox
        x_min = np.amin( corner[ :, 0 ] ); x_max = np.amax( corner[ :, 0 ] )
        y_min = np.amin( corner[ :, 1 ] ); y_max = np.amax( corner[ :, 1 ] )

        # check limits
        x_min_c = max( 0, x_min ); y_min_c = max( 0, y_min )
        x_max_c = min( x_max, dims[1] - 1 ); y_max_c = min( y_max, dims[0] - 1 )        

        area = ( x_max - x_min ) * ( y_max - y_min )
        area_c = ( x_max_c - x_min_c ) * ( y_max_c - y_min_c )

        # only retain bboxes not constrained by image edges
        if area_c / area > 0.95:

            record[ 'bbox' ] = [ x_min_c, y_min_c, x_max_c, y_max_c ]

            # readjust perimeter points
            corner[ :, 0 ] = np.where( corner[ :, 0 ] < 0.0, 0.0, corner[ :, 0 ] )
            corner[ :, 0 ] = np.where( corner[ :, 0 ] > dims[1] - 1, dims[1] - 1, corner[ :, 0 ] )

            corner[ :, 1 ] = np.where( corner[ :, 1 ] < 0.0, 0.0, corner[ :, 1 ] )
            corner[ :, 1 ] = np.where( corner[ :, 1 ] > dims[0] - 1, dims[0] - 1, corner[ :, 1 ] )

            # minimise distance between points
            d1 = np.linalg.norm( corner[ 1 ] - corner[ 2 ] ); d2 = np.linalg.norm( corner[ 1 ] - corner[ 3 ] )
            if d1 > d2:
                corner[ [ 2, 3 ] ] = corner[ [ 3, 2 ] ]

            record[ 'corner' ] = list( corner.flatten() )

        return record


    def rotatePoint( self, x, y, xc, yc, angle ):

        """
        compute rotation of point around origin
        """

        # Rotate point counterclockwise by a given angle around a given origin.
        qx = xc + math.cos(angle) * (x - xc) - math.sin(angle) * (y - yc)
        qy = yc + math.sin(angle) * (x - xc) + math.cos(angle) * (y - yc)

        return qx, qy


    def getSchema( self, s, records ):

        """
        convert annotation into ordered list for conversion into PASCAL VOC schema
        """

        # convert to PASCAL VOC annotation schema
        object_list = []
        for record in records:

            bbox = record[ 'bbox' ]; #corner = record[ 'corner' ]
            object_list.append( OrderedDict ( {     'name' : 'car',
                                                    'pose': 'Topdown',
                                                    'truncated' : 0,
                                                    'difficult': 0,
                                                    'bndbox': {'xmin': bbox[ 0 ], 'ymin': bbox[ 1 ], 'xmax': bbox[ 2 ], 'ymax': bbox[ 3 ] }
                                                    #'segmentation' : ','.join( (str(pt) for pt in corner ) ) 
                                            } ) )

        # return full schema as dictionary
        return OrderedDict ( {  'folder' : 'images',
                                'filename' : os.path.basename( s[ 'pathname' ] ),
                                'path' : os.path.dirname( s[ 'pathname' ] ),
                                'source' : { 'database': 'cowc' },
                                'size' : { 'width' : s[ 'width' ], 'height' : s[ 'height' ], 'depth' : 3 },
                                'segmented' : 0,
                                'items' : object_list } )


    def drawBoundingBoxes( self, pathname, records ):

        """
        placeholder
        """

        # no action if no bboxes
        if len ( records ) > 0:

            # load image
            img = cv2.imread( pathname )                                  
            height = img.shape[0]; width = img.shape[ 1 ]
                    
            # show image
            plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ); ax = plt.gca()
            fig = plt.gcf(); fig.canvas.set_window_title( os.path.basename( pathname ) )
            print ( pathname )

            # draw bbox lines
            colors = [ 'r', 'g', 'y', 'b', 'm', 'c' ]; idx = 0
            for record in records:    

                x0, y0, x1, y1 = record[ 'bbox' ]

                color = colors[ idx ] + '-'
                idx = idx + 1 if idx + 1 < len ( colors ) else 0

                ax.plot( [ x0, x1 ], [ y0, y0 ], color )
                ax.plot( [ x0, x1 ], [ y1, y1 ], color )
                ax.plot( [ x0, x0 ], [ y0, y1 ], color )
                ax.plot( [ x1, x1 ], [ y0, y1 ], color )

                """
                # get run length encoding from perimeter points string
                rl_encoding = mask.frPyObjects( [ record[ 'corner' ] ] , height, width )

                binary_mask = mask.decode( rl_encoding )
                binary_mask = np.amax(binary_mask, axis=2)

                masked = np.ma.masked_where(binary_mask == 0, binary_mask )
                ax.imshow( masked, 'jet', interpolation='None', alpha=0.5 )
                """

            plt.show()

        return
