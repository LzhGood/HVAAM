#coding=UTF-8
from __future__ import division
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, MyCallBack
from keras.layers import Input
from keras.models import Model
import os, cv2
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import hvaam, kl_divergence, correlation_coefficient, nss
def Mygenerator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        raise NotImplementedError
    images.sort()
    maps.sort()
    fixs.sort()
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))
    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian], [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)
        print(images[counter:counter + b_s])
def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':
        phase="train"

        x = Input((3, shape_r, shape_c))
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

        out_vector=hvaam([x, x_maps])
        m = Model(input=[x, x_maps], output=out_vector)
        print("Compiling HVAAM")

        m.compile(RMSprop(lr=1e-5), loss=[kl_divergence, correlation_coefficient, nss])
        m.load_weights('preTrainedWeights.pkl')
        if phase == 'train':
            if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
                print("Error")
                exit()
            print("Training HVAAM")
            m.fit_generator(Mygenerator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                            validation_data=Mygenerator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                            callbacks=[EarlyStopping(patience=3),
                                       ModelCheckpoint('weights.HVAAM.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True),MyCallBack()])
        elif phase == "test":
            # Output Folder Path
            output_folder = 'ouputImageList/'
            imgs_test_path='inputImageList/'

            file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            file_names.sort()
            nb_imgs_test = len(file_names)

            if nb_imgs_test % b_s != 0:
                print("Error")
                exit()

            print("Loading HVAAM weights")
            m.load_weights('HVAAM.pkl')

            print("Predicting saliency maps for " + imgs_test_path)
            predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0]
            for pred, name in zip(predictions, file_names):
                original_image = cv2.imread(imgs_test_path + name, 0)
                res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
                cv2.imwrite(output_folder + '%s' % name, res.astype(np.uint8))
        else:
            raise NotImplementedError
