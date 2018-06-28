#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import math


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    point_cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    outlier = point_cloud.make_statistical_outlier_filter()
    outlier.set_mean_k(32)
    outlier.set_std_dev_mul_thresh(0.5)
    outlier_point_cloud = outlier.filter()

    # TODO: Voxel Grid Downsampling
    LEAF_SIZE = 0.005
    vox = outlier_point_cloud.make_voxel_grid_filter()
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    outlier_vox_point_cloud = vox.filter()

    # TODO: PassThrough Filter
    passthrough = outlier_vox_point_cloud.make_passthrough_filter()
    passthrough.set_filter_field_name('z') # filter along this axis
    passthrough.set_filter_limits(0.6, 1.3)  # (axis_min, axis_max)
    filtered_point_cloud = passthrough.filter()

    # TODO: RANSAC Plane Segmentation
    seg = filtered_point_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01) # (max_distance)

    inliers, coefficients = seg.segment()


    # TODO: Extract inliers and outliers
    extracted_inlier = filtered_point_cloud.extract(inliers, negative=False) # table
    extracted_outliers = filtered_point_cloud.extract(inliers, negative=True) # objects

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(1050)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
            for i, indice in enumerate(indices):
                    color_cluster_point_list.append([white_cloud[indice][0],
                                                    white_cloud[indice][1],
                                                    white_cloud[indice][2],
                                                    rgb_to_float(cluster_color[j])])


    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    pcl_to_ros_cloud_objects =  pcl_to_ros(extracted_outliers)
    pcl_to_ros_cloud_table =  pcl_to_ros(extracted_inlier)
    pcl_to_ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    objects_pub.publish(pcl_to_ros_cloud_objects)
    table_pub.publish(pcl_to_ros_cloud_table)
    cluster_pub.publish(pcl_to_ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    objects_labels = []
    detected_objects= []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = extracted_outliers.extract(pts_list)
        cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(cluster, using_hsv=True)
        normals = get_normals(cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = sample_cloud
        detected_objects.append(do)

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    # Read the req/resp variables from "Grasp.srv" and "PickPlace.srv"
    object_name = String()
    test_scene_num = Int32()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    dict_list = []
    centroids = []

    object_dict = {}
    dropbox_dict = {}

    # TODO: Get/Read parameters
    objects_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    test_scene_num = 3

    # TODO: Parse parameters into individual variables
    for p in objects_param:
        object_dict[p['name']] = p['group']

    for p in dropbox_param:
        dropbox_dict[p['group']] = (p['name'], p['position'])

    # TODO: Rotate PR2 in place to capture side tables for the collision map



    # TODO: Loop through the pick list
    for object in objects_param:

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object
        pick_pose.position.x = 0
        pick_pose.position.y = 0
        pick_pose.position.z = 0

        pick_pose.orientation.x = 0
        pick_pose.orientation.y = 0
        pick_pose.orientation.z = 0
        pick_pose.orientation.w = 0

        place_pose.orientation.x = 0
        place_pose.orientation.y = 0
        place_pose.orientation.z = 0
        place_pose.orientation.w = 0

        for detected_object in object_list:
            if detected_object.label in object_dict:
                object_name.data = str(detected_object.label)

                centroid = np.mean(ros_to_pcl(detected_object.cloud).to_array(), axis=0)[:3]
                centroids.append(centroid)

                pick_pose.position.x = np.asscalar(centroid[0])
                pick_pose.position.y = np.asscalar(centroid[1])
                pick_pose.position.z = np.asscalar(centroid[2])

                if object['group'] == 'green':
                    arm_name.data = 'right'
                elif object['group'] == 'red':
                    arm_name.data = 'left'

                # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
                yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                dict_list.append(yaml_dict)

                break

        place_pose.position.x = dropbox_dict[arm_name.data][0]
        place_pose.position.y = dropbox_dict[arm_name.data][1]
        place_pose.position.z = dropbox_dict[arm_name.data][2]



        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_%s.yaml'.format(test_scene_num), dict_list)



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
