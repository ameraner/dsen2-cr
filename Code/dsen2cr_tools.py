import csv
import os
from random import shuffle

from dataIO import make_dir, DataGenerator, process_predicted
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import plot_model
from myCallbacks import CSV_NBatchLogger, NBatchLogger, TensorBoardWrapper


def train_dsen2cr(model, model_name, base_out_path, resume_file, train_filelist, val_filelist, lr, log_step_freq,
                  shuffle_train, data_augmentation, random_crop, batch_size, scale, clip_max, clip_min, max_val_sar,
                  use_cloud_mask, cloud_threshold, crop_size, epochs_nr, initial_epoch, input_data_folder, input_shape,
                  max_queue_size, use_multi_processing, workers):
    """Start or resume training of DSen2-CR model."""

    print('Training model name: {}'.format(model_name))

    out_path_train = make_dir(os.path.join(base_out_path, model_name, '/'))

    # generate model information and metadata
    plot_model(model, to_file=os.path.join(out_path_train, model_name + 'model.png'), show_shapes=True,
               show_layer_names=True)
    model_yaml = model.to_yaml()
    with open(out_path_train + model_name + "model.yaml", 'w') as yaml_file:
        yaml_file.write(model_yaml)
    print("Model information files created at ", out_path_train)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize callbacks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # instantiate model checkpoint callback
    model_filepath = os.path.join(out_path_train, model_name + '_{epoch:02d}-{val_loss:.4f}' + '.hdf5')
    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto')

    # instantiate csv logging callback
    csv_filepath = os.path.join(out_path_train, model_name + '_csvlog.csv')
    csv_logger = CSVLogger(csv_filepath, append=True, separator=";")
    csv_batch_logger = CSV_NBatchLogger(1, out_path_train, model_name, initial_epoch, separator=';')

    # instantiate NBatch logger
    batch_logger = NBatchLogger(log_step_freq, out_path_train, model_name, initial_epoch, lr)

    # instantiate Tensorboard logger
    # extract sample from validation dataset
    val_filelist_tensorboard = val_filelist
    shuffle(val_filelist_tensorboard)
    val_filelist_tensorboard = val_filelist_tensorboard[0:batch_size]

    params = {'input_dim': input_shape,
              'batch_size': batch_size,
              'shuffle': shuffle_train,
              'scale': scale,
              'include_target': True,
              'data_augmentation': False,
              'random_crop': False,
              'crop_size': crop_size,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'input_data_folder': input_data_folder,
              'use_cloud_mask': use_cloud_mask,
              'max_val_sar': max_val_sar,
              'cloud_threshold': cloud_threshold}
    val_tensorboard_generator = DataGenerator(val_filelist_tensorboard, **params)

    tensorboard = TensorBoardWrapper(val_tensorboard_generator, input_dim=input_shape, nb_steps=1,
                                     batch_size=batch_size, log_dir=out_path_train, histogram_freq=1,
                                     write_graph=False,
                                     batch_nr=batch_size, write_grads=True, update_freq=500,
                                     learning_phase=False)

    # define callbacks list
    callbacks_list = [checkpoint, csv_logger, batch_logger, csv_batch_logger, tensorboard]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params = {'input_dim': input_shape,
              'batch_size': batch_size,
              'shuffle': shuffle_train,
              'scale': scale,
              'include_target': True,
              'data_augmentation': data_augmentation,
              'random_crop': random_crop,
              'crop_size': crop_size,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'input_data_folder': input_data_folder,
              'use_cloud_mask': use_cloud_mask,
              'max_val_sar': max_val_sar,
              'cloud_threshold': cloud_threshold}
    training_generator = DataGenerator(train_filelist, **params)

    params = {'input_dim': input_shape,
              'batch_size': batch_size,
              'shuffle': shuffle_train,
              'scale': scale,
              'include_target': True,
              'data_augmentation': False,  # keep false
              'random_crop': False,
              'crop_size': crop_size,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'input_data_folder': input_data_folder,
              'use_cloud_mask': use_cloud_mask,
              'max_val_sar': max_val_sar,
              'cloud_threshold': cloud_threshold
              }

    validation_generator = DataGenerator(val_filelist, **params)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print('Training starts...')

    if resume_file is not None:
        print("Will resume from the weights in file {}".format(resume_file))
        model.load_model(resume_file)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs_nr,
                        verbose=1,
                        callbacks=callbacks_list,
                        shuffle=False,
                        initial_epoch=initial_epoch,
                        use_multiprocessing=use_multi_processing,
                        max_queue_size=max_queue_size,
                        workers=workers)


def predict_dsen2cr(predict_file, model, model_name, base_out_path, input_data_folder, predict_filelist, batch_size,
                    clip_min, clip_max, crop_size, input_shape, use_cloud_mask, cloud_threshold, max_val_sar,
                    scale):
    print("Predicting using file: {}".format(predict_file))
    print("Using this model: {}".format(model_name))

    # load the model weights at checkpoint
    model.load_weights(predict_file)

    out_path_predict = make_dir(os.path.join(base_out_path, model_name))
    predicted_images_path = make_dir(os.path.join(out_path_predict, 'images_output/'))

    print("Initializing generator for prediction and evaluation")
    params = {'input_dim': input_shape,
              'batch_size': 1,
              'shuffle': False,
              'scale': scale,
              'include_target': True,
              'data_augmentation': False,
              'random_crop': False,
              'crop_size': crop_size,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'input_data_folder': input_data_folder,
              'use_cloud_mask': use_cloud_mask,
              'max_val_sar': max_val_sar,
              'cloud_threshold': cloud_threshold}
    predict_generator = DataGenerator(predict_filelist, **params)

    eval_csv_name = out_path_predict + 'eval.csv'
    print("Storing evaluation metrics at ", eval_csv_name)

    with open(eval_csv_name, 'a') as eval_csv_fh:
        eval_writer = csv.writer(eval_csv_fh, dialect='excel')
        eval_writer.writerow(model.metrics_names)

        for i, (data, y) in enumerate(predict_generator):
            print("Processing file number ", i)
            # get evaluation metrics
            eval_results = model.test_on_batch(data, y)
            # write evaluation metrics
            eval_writer.writerow(eval_results)

            # predict output image
            predicted = model.predict_on_batch(data)
            # process predicted image
            process_predicted(predicted, predict_filelist[i * batch_size:i * batch_size + batch_size],
                              predicted_images_path,
                              scale, cloud_threshold, input_data_folder)

    print("Prediction finished with success!")
