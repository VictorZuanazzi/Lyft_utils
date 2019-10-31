import numpy as np
from tqdm import tqdm as tqdm
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


from LyftDataset import LyftDataset, LyftDatasetExplorer
from Box import Box
from PointCloud import PointCloud, RadarPointCloud, LidarPointCloud


# Data path
DATA_PATH = '../Datasets/Lyft-kaggle/'

# Initialize dataset
lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'train_data')

# select scene
my_scene = lyft_dataset.scene[0]
my_sample_token = my_scene["first_sample_token"]
my_sample = lyft_dataset.get('sample', my_sample_token)

# # render top view from lidar
# lyft_dataset.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=1)
#
# # render all camera views and lidar
# lyft_dataset.render_sample(my_sample_token)
#
# # render just one of the camera views
# sensor_channels = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
# sensor_channel = np.random.choice(sensor_channels)
# my_sample_data = lyft_dataset.get('sample_data', my_sample['data'][sensor_channel])
# lyft_dataset.render_sample_data(my_sample_data['token'])
#
# # render point cloud on the image
# lyft_dataset.render_pointcloud_in_image(sample_token = my_sample["token"],
#                                         dot_size = 1,
#                                         camera_channel = 'CAM_FRONT')
#
# # render views with an specefic anotation
# my_annotation_token = my_sample['anns'][10]
# my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)
# lyft_dataset.render_annotation(my_annotation_token)
#
# # renders (annotation's) instance
# my_instance = lyft_dataset.instance[100]
# instance_token = my_instance['token']
# lyft_dataset.render_instance(instance_token)  # instance rendering
# lyft_dataset.render_annotation(my_instance['last_annotation_token'])  # annotation rendering
#
# # render top lidar
# lyft_dataset.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5)
#
# plt.show()

from IPython.display import HTML

anim = lyft_dataset.animate_images(scene=0, frames=50, interval=1)
HTML(anim.to_jshtml(fps=8))