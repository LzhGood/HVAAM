# batch size
b_s = 10
# number of rows of input images
shape_r = 240#240
# number of cols of input images
shape_c = 320#320
# number of rows of downsampled maps
shape_r_gt = 30#30
# number of cols of downsampled maps
shape_c_gt = 40#40
# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
# final upsampling factor
upsampling_factor = 16#16
# number of epochs
nb_epoch = 12#10
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16
regular_size=1.0
regular_size_=2.5
lzh=0

# path of training images
imgs_train_path = 'D:/humanComputerInteraction/Train/Source/'
# path of training maps
maps_train_path = 'D:/humanComputerInteraction/Train/Saliency/'
# path of training fixation maps
fixs_train_path = 'D:/humanComputerInteraction/Train/Fixtion/'
# number of training images
nb_imgs_train = 500
# path of validation images
imgs_val_path = 'D:/humanComputerInteraction/Val/Source/'
# path of validation maps
maps_val_path = 'D:/humanComputerInteraction/Val/Saliency/'
# path of validation fixation maps
fixs_val_path = 'D:/humanComputerInteraction/Val/Fixtion/'
# number of validation images
nb_imgs_val = 125