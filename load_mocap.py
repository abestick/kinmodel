#!/usr/bin/env python
"""Loads data from C3D format files. This module provides a simple 
wrapper for the C3D-reading functionality of the Biomechanical Toolkit
(https://code.google.com/p/b-tk/). At present, it just provides methods 
to load a file, get basic metadata, and read individual samples or 
groups of samples.

In the future, the MocapFile class could be modified to support
mocap file formats other than C3D, such as BVH, CSV, etc.

Author: Aaron Bestick
"""
from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp
import scipy.linalg as spla
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
import sys
import os
#import OWL
from collections import deque
from Queue import Queue
from threading import Thread, RLock
import time

class MocapSource():
    __metaclass__=ABCMeta

    @abstractmethod
    def read(self, length=1, block=True):
        """Reads data from the underlying mocap source. By default, this method 
        will block until the data is read. Returns a tuple (frames, timestamps) 
        where frames is a (num_points, 3, length) ndarray of mocap points, and 
        timestamps is a (length,) ndarray of the timestamps, in seconds, of the 
        corresponding mocap points.

        Once the end of the file/stream is reached, calls to read() will return 
        None. If the end of the stream is reached before length frames are read,
        the returned arrays may have fewer elements than expected, and all 
        future calls to read() will return None.

        If called with block=False and no data is available, returns arrays with 
        length=0.

        Once the end of the file/stream is reached, calls to read() will return 
        None.
        """
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_num_points(self):
        pass

    @abstractmethod
    def get_length(self):
        pass

    @abstractmethod
    def get_framerate(self):
        pass

    @abstractmethod
    def set_sampling(self, num_samples, mode='uniform'):
        pass

    @abstractmethod
    def set_coordinates(self, markers, new_coords, mode='constant'):
        pass

    def __iter__(self):
        return MocapIterator(self)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class PhasespaceStream(MocapSource):
    def __init__(self, ip_address, num_points, framerate=None, buffer_length=2):
        import OWL
        self._num_points = num_points
        self._shutdown_flag = False
        self._start_time = 0
        self._frame_count = 0
        #Run the read loop at 2000Hz regardless of actual framerate to control 
        #jitter
        self._timer = RateTimer(2000)

        #Connect to the server
        OWL.owlInit(ip_address, 0)
        OWL.owlSetInteger(OWL.OWL_FRAME_BUFFER_SIZE, 0)

        tracker = 0
        OWL.owlTrackeri(tracker, OWL.OWL_CREATE, OWL.OWL_POINT_TRACKER)

        # add the points we want to track to the tracker
        for i in range(num_points):
            OWL.owlMarkeri(OWL.MARKER(tracker, i), OWL.OWL_SET_LED, i)

        #Activate tracker
        OWL.owlTracker(0, OWL.OWL_ENABLE)

        #Set frequency
        if framerate is None:
            OWL.owlSetFloat(OWL.OWL_FREQUENCY, OWL.OWL_MAX_FREQUENCY)
            # self._timer = RateTimer(1500)
        else:
            OWL.owlSetFloat(OWL.OWL_FREQUENCY, framerate)
            # self._timer = RateTimer(framerate*3)

        #Start streaming
        OWL.owlSetInteger(OWL.OWL_STREAMING, OWL.OWL_ENABLE)

        #Check for errors
        if OWL.owlGetError() != OWL.OWL_NO_ERROR:
            raise RuntimeError('An error occurred while connecting to the mocap server')

        #Initialize a circular read buffer
        self._read_buffer = _RingBuffer(buffer_length)

        #Start the reader thread
        self._reader = Thread(target=self._reader_thread)
        self._reader.daemon = True
        self._start_time = time.time()
        self._reader.start()

    def _reader_thread(self):
        while(1):
            self._timer.wait()
            markers = OWL.owlGetMarkers()
            if markers.size() > 0:
                #If there's data, add a frame to the buffer
                new_frame = np.empty((self._num_points, 3, 1))
                new_frame.fill(np.nan)

                #Add the markers
                for i in range(markers.size()):
                    m = markers[i]
                    if m.cond > 0:
                        new_frame[m.id,0,0] = m.x
                        new_frame[m.id,1,0] = m.y
                        new_frame[m.id,2,0] = m.z
                        # print("%d: %f %f %f" % (m.id, m.x, m.y, m.z))
                timestamp = np.array(time.time())
                self._read_buffer.put((new_frame, timestamp))
                self._frame_count += 1

            if OWL.owlGetError() != OWL.OWL_NO_ERROR:
                print('A mocap read error occurred')
            if self._shutdown_flag:
                return

    def read(self, length=1, block=True):
        """Reads data from the underlying mocap source. By default, this method 
        will block until the data is read. Returns a tuple (frames, timestamps) 
        where frames is a (num_points, 3, length) ndarray of mocap points, and 
        timestamps is a (length,) ndarray of the timestamps, in seconds, of the 
        corresponding mocap points.

        Once the end of the file/stream is reached, calls to read() will return 
        None. If the end of the stream is reached before length frames are read,
        the returned arrays may have fewer elements than expected, and all 
        future calls to read() will return None.

        If called with block=False and no data is available, returns arrays with 
        length=0.

        Once the end of the file/stream is reached, calls to read() will return 
        None.
        """
        # #Initialize an array to hold the data if length > 0
        # if length > 1:

        #     read_data = np.empty((self._num_points, 3, length))
        #     read_data[:] = np.nan

        # #Read the data
        # for i in range(length):
        #     raw_data = OWL.owlGetMarkers()
        #     for marker in raw_data:
        #         if marker.cond > 0:
        #             read_data[marker.id,0,i] = marker.x
        #             read_data[marker.id,1,i] = marker.y
        #             read_data[marker.id,2,i] = marker.z

        #     if OWL.owlGetError() != OWL.OWL_NO_ERROR:
        #         raise RuntimeError('An error occurred while reading data from the server')
        # return read_data
        frames = []
        timestamps = []
        for i in range(length):
            next_sample = self._read_buffer.get(block=True)
            frames.append(next_sample[0])
            timestamps.append(next_sample[1])
        return np.dstack(frames), np.hstack(timestamps)

    def close(self):
        self._shutdown_flag = True
        self._reader.join()
        OWL.owlDone()

    def get_num_points(self):
        return self._num_points

    def get_length(self):
        return -1

    def get_framerate(self):
        return self._frame_count / (time.time() - self._start_time)

    def set_sampling(self, num_samples, mode='uniform'):
        pass

    def set_coordinates(self, markers, new_coords, mode='constant'):
        pass


class MocapFile(MocapSource):
    def __init__(self, input_data):
        """Loads a motion capture file (currently a C3D file) to create 
        a new MocapFile object
        """
        import btk

        #Declare fields
        self._all_frames = None
        self._timestamps = None
        self._read_pointer = 0 #Next element that will be returned by read()
        
        #Check whether data is another MocapFile instance
        if hasattr(input_data, '_all_frames') and hasattr(input_data, '_timestamps'):
            self._all_frames = input_data._all_frames
            self._timestamps = input_data._timestamps
            
        #If not, treat data as a filepath to a C3D file
        else:
            #Initialize the file reader
            data = btk.btkAcquisitionFileReader()
            data.SetFilename(input_data)
            data.Update()
            data = data.GetOutput()
            
            #Get the number of markers tracked in the file
            num_points = 0
            run = True
            while run:
                try:
                    data.GetPoint(num_points)
                    num_points = num_points + 1
                except(RuntimeError):
                    run = False
            
            #Get the length of the file
            length = data.GetPointFrameNumber()
            
            #Load the file data into an array
            self._all_frames = sp.empty((num_points, 3, length))
            for i in range(num_points):
                self._all_frames[i,:,:] = data.GetPoint(i).GetValues().T
            
            #Replace occluded markers (all zeros) with NaNs
            norms = la.norm(self._all_frames, axis=1)
            ones = sp.ones(norms.shape)
            nans = ones * sp.nan
            occluded_mask = np.where(norms != 0.0, ones, nans)
            occluded_mask = np.expand_dims(occluded_mask, axis=1)
            self._all_frames = self._all_frames * occluded_mask
            
            #Calculate and save the timestamps
            frequency = data.GetPointFrequency()
            period = 1/frequency
            self._timestamps = sp.array(range(length), dtype='float') * period
            
            #Make the arrays read-only
            self._all_frames.flags.writeable = False
            self._timestamps.flags.writeable = False
    
    def read(self, length=1, block=True):
        #Make sure we don't try to read past the end of the file
        file_len = self.get_length()
        if file_len == self._read_pointer:
            return None
        elif length > file_len - self._read_pointer:
            length = file_len - self._read_pointer

        #Read the frames and timestamps
        frames = self.get_frames()[:,:,self._read_pointer:self._read_pointer+length+1]
        timestamps = self.get_timestamps()[self._read_pointer:self._read_pointer+length+1]

        #Increment the read pointer
        self._read_pointer = self._read_pointer + length
        return frames, timestamps

    def close(self):
        pass
    
    def get_frames(self):
        """Returns a (num_points, 3, length) array of the mocap points.
        Always access mocap data through this method.
        """
        return self._all_frames
    
    def get_timestamps(self):
        """Returns a 1-D array of the timestamp (in seconds) for each frame
        """
        return self._timestamps
    
    def get_num_points(self):
        """Returns the total number of points tracked in the mocap file
        """
        return self.get_frames().shape[0]
    
    def get_length(self):
        """Returns the total number of frames in the mocap file
        """
        return self.get_frames().shape[2]
    
    def get_framerate(self):
        """Returns the average framerate of the mocap file in Hz"""
        duration = self._timestamps[-1] - self._timestamps[0]
        framerate = (self.get_length() - 1) / duration
        return framerate
    
    def set_start_end(self, start, end):
        """Trims the mocap sequence to remove data before/after the
        specified start/end frames, respectively.
        """
        self._all_frames = self._all_frames[:,:,start:end+1]
        self._timestamps = self._timestamps[start:end+1]
        self._all_frames.flags.writeable = False
        self._timestamps.flags.writeable = False
        
    def set_sampling(self, num_samples, mode='uniform'):
        if mode == 'uniform':
            indices = sp.linspace(0, self.get_length()-1, num=num_samples)
            indices = sp.around(indices).astype(int)
        elif mode == 'random':
            indices = sp.random.randint(0, self.get_length(), num_samples)
            indices = sp.sort(indices)
        else:
            raise TypeError('A valid mode was not specified')
            
        self._all_frames = self._all_frames[:,:,indices]
        self._timestamps = self._timestamps[indices]
        self._all_frames.flags.writeable = False
        self._timestamps.flags.writeable = False
    
    def plot_frame(self, frame_num, mark_num):
        """Plots the location of each marker in the specified frame
        """
        #Get the frame
        frame = self.get_frames()[:,:,frame_num]
        xs = frame[:,0]
        ys = frame[:,1]
        zs = frame[:,2]
        
        #Make the plot
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
        markers = ['r']*self.get_num_points()
        markers[mark_num] = 'b'
        axes.scatter(xs, ys, zs, c=markers, marker='o')
        axes.auto_scale_xyz([-1000,1000], [-1000, 1000], [0, 2000])
        axes.set_xlabel('X Label')
        axes.set_ylabel('Y Label')
        axes.set_zlabel('Z Label')
        axes.set_zlabel('Z Label')
    
    def set_coordinates(self, markers, new_coords, mode='constant'):
        """Changes the coordinate system for the mocap sequence from the
        original coordinate frame x to a new coordinate frame y. The 
        markers argument specifies the stationary mocap markers to compute 
        the coordinate change based on, and the coords_y argument is a 
        (len(markers), 3) array of the desired positions of the specified
        markers in the new y frame.
        """
        trans_points = None
        if mode == 'constant':
            #Compute the average coordinates of each marker over all samples,
            #ignoring occluded markers
            orig_coords = np.nanmean(self.get_frames()[markers,:,:], axis=2)
            
            #Compute and apply the transformation
            homog = find_homog_trans(orig_coords, new_coords)[0]
            orig_points = np.hstack((self.get_frames(),
                    sp.ones((self.get_frames().shape[0], 1, self.get_frames().shape[2]))))
            trans_points = np.transpose(np.dot(homog, orig_points.T), axes=[2,0,1])[:,0:3,:]

        elif mode == 'time_varying':
            #Iterate over each frame
            markers = np.array(markers)
            rot_0 = np.ones(6)
            homog_0 = np.eye(4)
            trans_points = self.get_frames().copy()
            for i in range(trans_points.shape[2]):
                #Find which of the specified markers are visible in this frame
                visible_inds = np.where(~np.isnan(trans_points[markers,0,i]))[0]

                #Just apply the last transformation if no markers are visible
                homog=None
                if len(visible_inds) == 0:
                    homog = homog_0

                #Otherwise, compute the transformation
                else:
                    orig_points = trans_points[markers[visible_inds],:,i]
                    desired_points = new_coords[visible_inds]
                    homog, rot_0 = find_homog_trans(orig_points, desired_points, rot_0=rot_0)
                    homog_0 = homog

                #Apply the transformation to the frame
                homog_coords = np.vstack((trans_points[:,:,i].T, np.ones((1,trans_points.shape[0]))))
                homog_coords = np.dot(homog, homog_coords)
                trans_points[:,:,i] = homog_coords.T[:,0:3]
                print('Applying base frame transformation to frame: ' + str(i) + '/' + str(trans_points.shape[2]),
                end='\r')
                sys.stdout.flush()
            print()

        else:
            raise TypeError('The specified mode is invalid')

        #Save the transformed points
        self._all_frames = trans_points
        self._all_frames.flags.writeable = False
    
    def __iter__(self):
        return MocapIterator(self)

class MocapArray(MocapFile):
    def __init__(self, array, framerate):
        if array.shape[1] != 3 or array.ndim != 3:
            raise TypeError('Input array is not the correct shape')

        self._all_frames = array
        self._timestamps = np.array(range(array.shape[2])) * (1.0/framerate)
        self._read_pointer = 0 #Next element that will be returned by read()

class MocapIterator():
    def __init__(self, mocap_obj):
        #Check that mocap_obj is a MocapFile instance
        if not hasattr(mocap_obj, 'read'):
            raise TypeError('A valid MocapSource instance was not given')
        
        #Define fields
        self.mocap_obj = mocap_obj
        # self.mocap_data = mocap_obj.get_frames()
        # self.mocap_time = mocap_obj.get_timestamps()
        # self.current_frame = 0
    
    def __iter__(self):
        return self

    def next(self):
        value = self.mocap_obj.read()
        if value is not None:
            return value
        else:
            raise StopIteration()
    
    # def next(self):
    #     if self.current_frame >= self.mocap_data.shape[2]:
    #         raise StopIteration()
    #     else:
    #         self.current_frame = self.current_frame + 1
    #         return self.mocap_data[:,:,self.current_frame-1], self.mocap_time[self.current_frame-1]

class _RingBuffer():
    def __init__(self, size):
        self._buffer = Queue(maxsize=size)

    def put(self, item):
        if self._buffer.full():
            #If the buffer is full, discard the oldest element to make room
            self._buffer.get()
        self._buffer.put(item)

    def get(self, block=True):
        return self._buffer.get(block=block)

class RateTimer():
    def __init__(self, frequency):
        self._loop_time = 1.0 / frequency
        self._next_time = None

    def wait(self):
        if self._next_time is None:
            self._next_time = time.time() + self._loop_time
        else:
            wait_time = self._next_time - time.time()
            self._next_time += self._loop_time
            if wait_time > 0:
                # print('Waiting: ' + str(wait_time))
                time.sleep(wait_time)


def find_homog_trans(points_a, points_b, err_threshold=0, rot_0=None):
    """Finds a homogeneous transformation matrix that, when applied to 
    the points in points_a, minimizes the squared Euclidean distance 
    between the transformed points and the corresponding points in 
    points_b. Both points_a and points_b are (n, 3) arrays.
    """
    #OLD ALGORITHM ----------------------
    #Align the centroids of the two point clouds
    cent_a = sp.average(points_a, axis=0)
    cent_b = sp.average(points_b, axis=0)
    points_a = points_a - cent_a
    points_b = points_b - cent_b
    
    #Define the error as a function of a rotation vector in R^3
    rot_cost = lambda rot: (sp.dot(vec_to_rot(rot), points_a.T).T
                    - points_b).flatten()**2
    
    #Run the optimization
    if rot_0 == None:
        rot_0 = sp.zeros(3)
    rot = opt.leastsq(rot_cost, rot_0)[0]
    
    #Compute the final homogeneous transformation matrix
    homog_1 = sp.eye(4)
    homog_1[0:3, 3] = -cent_a
    homog_2 = sp.eye(4)
    homog_2[0:3,0:3] = vec_to_rot(rot)
    homog_3 = sp.eye(4)
    homog_3[0:3,3] = cent_b
    homog = sp.dot(homog_3, sp.dot(homog_2, homog_1))
    return homog, rot
    #---------------------------------------


    # #Define the error function
    # def error(state):
    #     rot = state[0:3]
    #     trans = state[3:6]

    #     #Construct a homography matrix
    #     homog = np.eye(4)
    #     homog[0:3,3] = trans
    #     homog[0:3,0:3] = vec_to_rot(rot)

    #     #Transform points_a
    #     points_a_h = np.hstack((points_a, np.ones((points_a.shape[0],1))))
    #     trans_points_a = homog.dot(points_a_h.T).T[:,0:3]

    #     #Compute the error
    #     err = la.norm(points_a - trans_points_a, axis=1)**2
    #     return err
    
    # #Run the optimization
    # if rot_0 == None:
    #     rot_0 = sp.zeros(6)
    # rot = opt.leastsq(error, rot_0, ftol=1e-20)[0]
    
    # #Compute the final homogeneous transformation matrix
    # homog = np.eye(4)
    # homog[0:3,3] = rot[3:6]
    # homog[0:3,0:3] = vec_to_rot(rot[0:3])
    # return homog, rot

def vec_to_rot(x):
    #Make a 3x3 skew-symmetric matrix
    skew = sp.zeros((3,3))
    skew[1,0] = x[2]
    skew[0,1] = -x[2]
    skew[2,0] = -x[1]
    skew[0,2] = x[1]
    skew[2,1] = x[0]
    skew[1,2] = -x[0]
    
    #Compute the rotation matrix
    rot = spla.expm(skew)
    return rot
