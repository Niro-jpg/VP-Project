# cell dimension
S = 8
# number of proposals
B = 2
# number of classes
C = 3
# learning rate
lr = 0.0001
# pre processing dimension
image_resize_dim = 448

model_path = "./model.pt"
folder_path = "./archive/train_zip/train"
test_folder_path = "./../Yolo/archive/test_zip/test"
#max dataset dimension
max_images_dim = 1000
batch_size = 32
#loss hyperparameters
lamda_obj = 5
lambda_noobj = 0.5
#proposal thresehold
epsilon = 0.7
epochs = 60