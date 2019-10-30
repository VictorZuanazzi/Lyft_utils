import numpy as np
from tqdm import tqdm as tqdm
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


from LyftDataset import LyftDataset, LyftDatasetExplorer
from Box import Box
from PointCloud import PointCloud, RadarPointCloud, LidarPointCloud

def generate_next_token(scene):
    scene = lyft_dataset.scene[scene]
    sample_token = scene['first_sample_token']
    sample_record = lyft_dataset.get("sample", sample_token)

    while sample_record['next']:
        sample_token = sample_record['next']
        sample_record = lyft_dataset.get("sample", sample_token)

        yield sample_token

DATA_PATH = '../Datasets/Lyft-kaggle/'

lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'train_data')

my_scene = lyft_dataset.scene[0]
my_sample_token = my_scene["first_sample_token"]
my_sample = lyft_dataset.get('sample', my_sample_token)
lyft_dataset.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=1)
plt.show()


