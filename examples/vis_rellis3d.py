
import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

# construct a dataset by specifying dataset_path
#dataset = ml3d.datasets.SemanticKITTI(dataset_path='/home/sa001/workspace/dataset/semantic_kitti/')
#dataset = ml3d.datasets.SemanticKITTI(dataset_path='/media/sa001/programfox2TB/dataset/RELLIS-3D/dataset_semantickitti_compatible/')
dataset = ml3d.datasets.Rellis3D(dataset_path='/media/sa001/programfox2TB/dataset/RELLIS-3D/', test_split=[ '02' ],
    training_split=[ '00', '01', '03', '04'],
    validation_split=['01'],
    all_split=[ '00', '01', '02', '03', '04' ])

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('validation')
#all_split = dataset.get_split('training')

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, 'validation', indices=range(100)) #9409
