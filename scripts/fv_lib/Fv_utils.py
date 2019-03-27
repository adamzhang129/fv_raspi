#! /usr/bin/env pyhton


from undistort_crop_resize import *
import cv2
from sklearn.neighbors import KDTree
import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)

from scipy import ndimage
from skimage import morphology, util, filters
import skimage
from scipy.interpolate import Rbf, griddata
import time
from pynhhd.nHHD import nHHD
from yaml import load

from scipy.spatial import cKDTree

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


class Fv:
    def __init__(self, vid=0, interp_nx=30, interp_ny=30):
        if isinstance(vid, int):
            print('Starting video captrue with webcam.')
            self.cap = cv2.VideoCapture(vid)
            self.map1, self.map2 = undistort_setup()
            
        else:
            print('Streaming with image topic from raspberry pi.')
            self.bridge = CvBridge()
            self.sub_img = rospy.Subscriber(vid, CompressedImage, self.image_cb)
            self.cap = None
            self.image = []
        
        self.sub_dataset_saving_command = rospy.Subscriber('/FV_l/dataset_saving_command', String, self.command_cb)
        self.pub_img = rospy.Publisher('/FV_l/disp_vector', CompressedImage)
        self.pub_disp = rospy.Publisher('/FV_l/disp_optical_flow', Image)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = 1
        params.blobColor = 0
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 60

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)

        # params for interpolation and decomposition

        # y_min = 20
        # y_max = 480 - 20
        # x_min = 120 - 80
        # x_max = 520 + 80
        y_min = 0
        y_max = 480
        x_min = 0
        x_max = 640
        self.ROI = [y_min, y_max, x_min, x_max]
        self.width, self.height = x_max - x_min, y_max - y_min

        self.Nx, self.Ny = interp_nx, interp_ny
        # since computation is demanding for rbf interpolation method, we switch to griddata cubic method
        # griddata doesn't extropolate outside of convex hull of known scatters (outside values would be NAN),
        # thus we need to shrink the region to be within the convex hull of data (manual selected)
        # ranges for griddata interpolation
        x_i_min = 3*self.width/(self.Ny-1)-20
        x_i_max = self.width - x_i_min
        y_i_min = 6*self.height/(self.Ny-1)-25
        y_i_max = self.height - y_i_min
        
        x = np.linspace(x_i_min, x_i_max, self.Nx)
        y = np.linspace(y_i_min, y_i_max, self.Ny)

        self.X, self.Y = np.array(np.meshgrid(x, y))
        self.XX = self.X.ravel()  # flatten
        self.YY = self.Y.ravel()

        # decomposition object
        dx = float(self.width - 2*x_i_min) / (self.Nx - 1)
        dy = float(self.height - 2*y_i_min) / (self.Ny - 1)
        grid = (self.Nx, self.Ny)

        # self.decomp_obj = nHHD(grid=grid, spacings=(dy, dx))

        self.loc_0 = []
        self.recent_loc = []
        self.count = 0
        self.vfield = []

        self.v_sum_mag = None
        self.v_sum_ang = None
        self.d_mag_sum = None
        self.sum_torque = None
        
        #########################################
        #   variables relevant to data collection
        #########################################
        self.N_frames = 30
        self.disp_mat  = np.zeros([self.Nx * self.Ny *2, self.N_frames])
        
        self.disp_field = []
        self.dataset_count = [0,0,0,0] # 4 kinds of classes

    def image_cb(self, img_msg):
        try:
            # print img_msg.header.stamp
            # self.image = self.bridge.imgmsg_to_cv2(img_msg)
            np_arr = np.fromstring(img_msg.data, np.uint8)
            self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # print self.image.shape
        except CvBridgeError as e:
            print(e)
        # print('msg recieved...........')
        self.track(time_verbose=False)
        # cv2.imshow('raw_image', self.image)
        # cv2.waitKey(3)
    
    def command_cb(self, msg):
        key = msg.data
        dataset_path = '../dataset'
        
        self.dataset_count = save_data(self.disp_mat, key, dataset_path, self.dataset_count)

    def resize(self, img, ratio):
        self.width = self.width*ratio
        self.height = self.height*ratio
        return cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

    def crop(self, img):
        return img[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]]

    def blob_detect(self, img):
        keypoints = self.detector.detect(img)

        # print(len(keypoints))
        locs = []
        for i in range(0, len(keypoints)):
            locs.append([keypoints[i].pt[0], keypoints[i].pt[1]])

        # print(np.array(locs))
        return np.array(locs), keypoints


    def track(self, time_verbose=0):

        start = time.time()

        if not self.cap == None:
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # only when the frame is captured from webcam (USB interface with fisheye lens)
            # the frame need to be undistorted.
            gray_ud = cv2.remap(gray, self.map1,
                                self.map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        else:
            frame = self.image # the image from rostopic raspicam_node msg
            # print '...'
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_ud = gray
            
        if time_verbose == 1:
            print 'time interval (wait for frames capturing): {}'.format(time.time() - start)
        
        gray_crop = self.crop(gray_ud)
        # print 'time interval -1: {}'.format(time.time() - start)
        # gray_resize = self.resize(gray_crop, 0.5)
        # cv2.imshow('resized', gray_resize)
        loc, keypoints = self.blob_detect(gray_crop)
        # print len(keypoints)
        if time_verbose == 1:
            print 'time till (mainly blob detection): {}'.format(time.time() - start)

        if self.count == 0:

            self.loc_0 = loc.copy()
            self.recent_loc = loc.copy()
        elif self.count > 0:
            # print('============frame {}================'.format(count))
            # print(loc_0[1,:])
            kdt = KDTree(loc, leaf_size=30, metric='euclidean')
            # kdt = cKDTree(loc, leafsize=30)
            dist, ind = kdt.query(self.recent_loc, k=1)
            thd = (dist < 14) * 1
            thd_nz = np.where(thd)[0]
            # update point if close enough point are detected
            self.recent_loc[thd_nz] = np.reshape(loc[ind[thd_nz]], (len(thd_nz), 2))

            # visualize the displacement field
            loc_v = 2 * self.recent_loc - self.loc_0  # diff vector


            # interpolation
            disp = self.recent_loc - self.loc_0

            dx, dy = disp[:, 0], disp[:, 1]
            x, y = self.recent_loc[:, 0], self.recent_loc[:, 1]
            # if time_verbose == 1:
                # print 'time interval 1: {}'.format(time.time() - start)

            # interpolation_x = Rbf(x, y, dx)
            # interpolation_y = Rbf(x, y, dy)
            #
            # dx_interp = interpolation_x(self.XX, self.YY)
            # dy_interp = interpolation_y(self.XX, self.YY)

            # griddata interpolation
            dx_interp = griddata(self.recent_loc, dx, (self.XX, self.YY), method='cubic')
            dy_interp = griddata(self.recent_loc, dy, (self.XX, self.YY), method='cubic')
            # print dx_interp.shape

            # mag = np.sqrt(dx_interp ** 2 + dy_interp ** 2)
            if time_verbose == 1:
                print 'time till interpolation: {}'.format(time.time() - start)

            self.vfield = np.stack((dx_interp.reshape(self.Nx, self.Ny),
                                    dy_interp.reshape(self.Nx, self.Ny)), axis=2)
            # print self.vfield.shape
            self.data_collection(dx_interp, dy_interp, n_frames=self.N_frames)

            # img_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
            # # draw image and save vectors
            # for i in range(0, len(dx_interp)):
            #     cv2.arrowedLine(img_rgb, (int(np.around(self.XX[i])), int(np.around(self.YY[i]))),
            #                     (int(np.around(self.XX[i] + dx_interp[i])), int(np.around(self.YY[i] + dy_interp[i]))),
            #                     (0, 255, 255), thickness=2, tipLength=0.5)
            
            self.disp_field = vec_color_encoding(self.vfield[..., 0], self.vfield[..., 1], encoding='rgb')

            #### Create CompressedIamge ####
            # msg_vec = CompressedImage()
            # msg_vec.header.stamp = rospy.Time.now()
            # msg_vec.format = "jpeg"
            # msg_vec.data = np.array(cv2.imencode('.jpg', img_rgb)[1]).tostring()
            
            #----------another one---------------
            # msg_of = CompressedImage()
            # msg_of.header.stamp = rospy.Time.now()
            # msg_of.format = 'jpeg'
            # msg_of.data = np.array(cv2.imencode('.jpg', self.disp_field)[1]).tostring()

            # self.pub_img.publish(msg_vec)
            try:
                # self.pub_img.publish(self.bridge.cv2_to_imgmsg(img_rgb, 'bgr8'))
                self.pub_disp.publish(self.bridge.cv2_to_imgmsg(self.disp_field, 'bgr8'))
            
            # self.pub_disp.publish(msg_of)
            except CvBridgeError as e:
                print(e)
            
            if time_verbose == 1:
                print 'total time for tracking: {}'.format(time.time() - start)
            
            
            
            
            
        self.count += 1

    def tracking_reset(self):
        self.recent_loc = []
        self.loc_0 = []
        self.count = 0
    
    def data_collection(self, dx, dy, n_frames=30):
        if self.count < n_frames+1:
            self.disp_mat[:, self.count-1] = np.concatenate((dx, dy), axis=0)
        else:
            self.disp_mat[:, :-1] = self.disp_mat[:, 1:]
            self.disp_mat[:, -1] = np.concatenate((dx, dy), axis=0)
        
        if self.count == 30:
            print '[Data Collection] You can start saving data, the matrix frame is full'
            
        
        
        

    def wrench_estimate(self):
        # ======calculate tangential force =========================
        # print self.vfield

        vx_sum = np.sum(self.vfield[:, :, 0])
        vy_sum = np.sum(self.vfield[:, :, 1])

        self.v_sum_mag = np.hypot(vx_sum, vy_sum)
        self.v_sum_ang = np.arctan2(vy_sum, vx_sum)

        # ======calculate normal  ===============
        self.decomp_obj.decompose(self.vfield, verbose=0)
        d = self.decomp_obj.d
        r = self.decomp_obj.r
        h = self.decomp_obj.h

        d = d.reshape(self.Nx * self.Ny, 2)
        r = r.reshape(self.Nx * self.Ny, 2)
        h = h.reshape(self.Nx * self.Ny, 2)

        mag_d = np.hypot(np.abs(d[:, 0]), np.abs(d[:, 1]))
        self.d_mag_sum = np.sum(mag_d, axis=0)
        # print 'total magnitude of d: {}'.format(sum_d)

        # ====== calculate torsional force =========================

        # calculate the index of maximum and minimum of potential field as rotation center
        R = self.decomp_obj.nRu
        R_max_ind = np.unravel_index(np.argmax(R), R.shape)
        R_max_loc = (self.X[R_max_ind], self.Y[R_max_ind])

        R_min_ind = np.unravel_index(np.argmin(R), R.shape)
        R_min_loc = (self.X[R_min_ind], self.Y[R_min_ind])

        # validate using first derivative
        i_max, j_max = R_max_ind
        i_min, j_min = R_min_ind
        # validate with finite difference symbols
        if i_max > 0 and i_max < self.Nx-1 and j_max > 0 and j_max < self.Ny-1:
            if (R[i_max, j_max] - R[i_max - 1, j_max]) > 0 and (R[i_max + 1, j_max] - R[i_max, j_max])< 0 \
                    and (R[i_max, j_max] - R[i_max, j_max - 1]) > 0 and (R[i_max, j_max + 1] - R[i_max, j_max]) < 0:
                    # this point is the local maxima
                    # print 'max for R exist'
                    max_loc = R_max_loc
            else:
                max_loc = None
        else:
            max_loc = None

        if i_min > 0 and i_min < self.Nx-1 and j_min > 0 and j_min < self.Nx-1:
            if (R[i_min, j_min] - R[i_min - 1, j_min]) < 0 and (R[i_min + 1, j_min] - R[i_min, j_min]) > 0 \
                    and (R[i_min, j_min] - R[i_min, j_min - 1]) < 0 and (R[i_min, j_min + 1] - R[i_min, j_min]) > 0:
                    # this point is the local minima
                    # print 'min for R exist'
                    min_loc = R_min_loc
            else:
                min_loc = None
        else:
            min_loc = None

        XY = np.stack((self.XX, self.YY), axis=1)
        torque_max = 0.0
        if max_loc != None:
            cor_diff = XY - R_max_loc
            torque = np.cross(cor_diff, r)
            torque_max = np.sum(torque, axis=0)

        torque_min = 0.0
        if min_loc != None:
            cor_diff = XY - R_min_loc
            torque = np.cross(cor_diff, r)
            torque_min = np.sum(torque, axis=0)

        self.sum_torque = (torque_max + torque_min) / (self.Nx * self.Ny)
        # print self.v_sum_mag, self.a_tan, self.b_tan
        self.F_tan = self.inv_linear_func(self.v_sum_mag, self.a_tan, self.b_tan)
        self.F_nor = self.inv_linear_func(self.d_mag_sum, self.a_nor, self.b_nor)
        self.F_tor = self.inv_linear_func(self.sum_torque, self.a_tor, self.b_tor)

        F_tan_x = self.F_tan*np.cos(self.v_sum_ang)
        F_tan_y = self.F_tan*np.sin(self.v_sum_ang)

        return F_tan_x, F_tan_y, self.F_nor, self.F_tor

    def load_yaml(self, path):
        params_path = file(path)

        params = load(params_path)

        self.a_tan = params['fittings']['tangential']['a']
        self.b_tan = params['fittings']['tangential']['b']
        self.a_nor = params['fittings']['normal']['a']
        self.b_nor = params['fittings']['normal']['b']
        self.a_tor = params['fittings']['torsional']['a']
        self.b_tor = params['fittings']['torsional']['b']

    def inv_linear_func(self, y, a, b):
        return (y - b) / a



def vec_color_encoding(x, y, encoding='hsv'):
    if not x.shape == y.shape:
        print '2d vector components should have same shapes.'
        return None
    hsv = np.zeros((x.shape[0], x.shape[1], 3))
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(x, y)
    
    # print np.max(mag), np.min(mag)
    hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag*7
    
    hsv = np.uint8(hsv)
    if encoding == 'hsv':
        return hsv
    elif encoding == 'rgb':
        # print('converted')
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return bgr


import pandas as pd
import os

def save_data(data,command, path, count):
    
    # start = time.time()
    df = pd.DataFrame(data, columns=range(0, data.shape[1]))
    # print(df.head(5))
    
    
    k = command
    if k == 't':
        filename = os.path.join(path, '0', str(count[0]))
        count[0] += 1
        print('Saving the dataframe with translational slip label as {}'.format(filename))
    elif k == 'r':
        filename = os.path.join(path, '1', str(count[1]))
        count[1] += 1
        print('Saving the dataframe with ratational slip label as {}'.format(filename))
    elif k == 'l':
        filename = os.path.join(path, '2', str(count[2]))
        count[2] += 1
        print('Saving the dataframe with surface rolling label as {}'.format(filename))
    elif k == 's':
        filename = os.path.join(path, '3', str(count[3]))
        count[3] += 1
        print('Saving the dataframe with stable label as {}'.format(filename))
    
    # print 'time elapsed: {}'.format(time.time() - start)
    return count
        
import sys
import signal

def signal_handler(signal, frame):
    print("\nprogram exiting...")
    sys.exit(0)


if __name__ == "__main__":
    try:
        fv = Fv('/raspicam_node_l/image/compressed')
        # fv.load_yaml('./fitting_param.yaml')
        rospy.init_node('fv_node', anonymous=True)
    
        rate = rospy.Rate(30)
        
        dataset_count = [0, 0, 0, 0]

        while True:
            # if fv.count >= 2:
            #     # print(fv.disp_field)
            #     cv2.imshow('optical flow image', fv.disp_field)
            #
            #     ###################################
            #     #        dataset collection
            #     dataset_path = '../dataset'
            #
            #     dataset_count = save_data(fv.disp_mat, dataset_path, dataset_count)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                    
            signal.signal(signal.SIGINT, signal_handler)
            
            # rospy.spin()
    except rospy.ROSInterruptException:
        pass

    # fv.cap.release()
    cv2.destroyAllWindows()