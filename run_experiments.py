from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf2
import random as rn
import os
import time
import json
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import keras
from keras import backend as K
from keras.layers import Input, Reshape, Flatten, Dense, BatchNormalization, Activation, Embedding, GlobalMaxPooling1D, \
    Dropout
from keras.layers.convolutional import Conv1D, Conv1DTranspose
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from emetrics import get_rm2
from sklearn.metrics import average_precision_score
from arguments import argparser, logging
from dataset import DataGenerator
from datahelper import *

np.random.seed(1)
rn.seed(1)

session_conf = tf2.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf2.set_random_seed(0)
sess = tf2.Session(graph=tf2.get_default_graph(), config=session_conf)
K.set_session(sess)

sns.set_theme(style='white')
figdir = "figures/"


def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier  # this is from global space - change this as you need
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def ganForDrug(XD):
    def build_gen(zdim):
        model = Sequential()
        model.add(Dense(256 * 5, input_dim=zdim))
        model.add(Reshape((5, 256)))

        model.add(Conv1DTranspose(128, activation='relu', kernel_size=3, strides=5,
                                  padding='same'))
        model.add(BatchNormalization())

        model.add(Conv1DTranspose(64, activation='relu', kernel_size=3, strides=4,
                                  padding='same'))
        model.add(BatchNormalization())

        model.add(Conv1DTranspose(1, kernel_size=3, strides=2,
                                  padding='same'))
        model.add(Activation('tanh'))
        return model

    def build_dis(img_shape):
        model = Sequential()
        model.add(Input(shape=(200,)))
        model.add(Reshape((200, 1)))

        model.add(Conv1D(4, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(8, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(16, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(32, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(64, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))
        return model

    def build_gan(gen, dis):
        model = Sequential()
        model.add(gen)
        model.add(dis)
        return model

    def train(iterations, batch_size, interval, Xtrain):
        Xtrain = np.asarray(Xtrain)
        Xtrain = Xtrain / 32.5 - 1.0

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            ids = np.random.randint(0, Xtrain.shape[0], batch_size)
            imgs = Xtrain[ids]

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = gen_v.predict(z)
            gen_imgs = gen_imgs[:, :, 0]

            dloss_real = dis_v.train_on_batch(imgs, real)
            dloss_fake = dis_v.train_on_batch(gen_imgs, fake)

            dloss, accuracy = 0.5 * np.add(dloss_real, dloss_fake)
            z = np.random.normal(0, 1, (batch_size, 100))
            gloss = gan_v.train_on_batch(z, real)

            if (iteration + 1) % interval == 0:
                losses.append((dloss, gloss))
                accuracies.append(100.0 * accuracy)
                iteration_checks.append(iteration + 1)
                print("%d [D loss: %f , acc: %.2f] [G loss: %f]" %
                      (iteration + 1, dloss, 100.0 * accuracy, gloss))

        return dis_v

    zdim = 100

    dis_v = build_dis((200, 1))
    dis_v.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    gen_v = build_gen(zdim)
    dis_v.trainable = False
    gan_v = build_gan(gen_v, dis_v)
    gan_v.compile(loss='binary_crossentropy',
                  optimizer=Adam()
                  )

    losses = []
    accuracies = []
    iteration_checks = []

    dis_v = train(5000, 5, 500, XD)
    return dis_v


def ganForTarget(XT):
    def build_gen(zdim):
        model = Sequential()
        model.add(Dense(256 * 10, input_dim=zdim))
        model.add(Reshape((10, 256)))

        model.add(Conv1DTranspose(128, activation='relu', kernel_size=3, strides=5,
                                  padding='same'))
        model.add(BatchNormalization())

        model.add(Conv1DTranspose(64, activation='relu', kernel_size=3, strides=10,
                                  padding='same'))
        model.add(BatchNormalization())

        model.add(Conv1DTranspose(1, kernel_size=3, strides=4,
                                  padding='same'))
        model.add(Activation('tanh'))
        return model

    def build_dis(img_shape):

        model = Sequential()
        model.add(Input(shape=(2000,)))
        model.add(Reshape((2000, 1)))

        model.add(Conv1D(4, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(8, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(16, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(32, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(64, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))
        return model

    def build_gan(gen, dis):
        model = Sequential()
        model.add(gen)
        model.add(dis)
        return model

    def train(iterations, batch_size, interval, Xtrain):

        Xtrain = np.asarray(Xtrain)
        Xtrain = Xtrain / 12.5 - 1.0

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            ids = np.random.randint(0, Xtrain.shape[0], batch_size)
            imgs = Xtrain[ids]

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = gen_v.predict(z)
            gen_imgs = gen_imgs[:, :, 0]

            dloss_real = dis_v.train_on_batch(imgs, real)
            dloss_fake = dis_v.train_on_batch(gen_imgs, fake)
            dloss, accuracy = 0.5 * np.add(dloss_real, dloss_fake)

            z = np.random.normal(0, 1, (batch_size, 100))
            gloss = gan_v.train_on_batch(z, real)

            if (iteration + 1) % interval == 0:
                losses.append((dloss, gloss))
                accuracies.append(100.0 * accuracy)
                iteration_checks.append(iteration + 1)
                print("%d [D loss: %f , acc: %.2f] [G loss: %f]" %
                      (iteration + 1, dloss, 100.0 * accuracy, gloss))

        return dis_v

    zdim = 100

    dis_v = build_dis((2000, 1))
    dis_v.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    gen_v = build_gen(zdim)
    dis_v.trainable = False
    gan_v = build_gan(gen_v, dis_v)
    gan_v.compile(loss='binary_crossentropy',
                  optimizer=Adam()
                  )

    losses = []
    accuracies = []
    iteration_checks = []

    dis_v = train(5000, 10, 500, XT)
    return dis_v


def ganForBlosumTarget(XT):
    def build_gen(zdim):
        model = Sequential()
        model.add(Dense(256 * 10, input_dim=zdim))
        model.add(Reshape((10, 256)))

        model.add(Conv1DTranspose(128, activation='relu', kernel_size=3, strides=5,
                                  padding='same'))
        model.add(BatchNormalization())

        model.add(Conv1DTranspose(64, activation='relu', kernel_size=3, strides=10,
                                  padding='same'))
        model.add(BatchNormalization())

        model.add(Conv1DTranspose(20, kernel_size=3, strides=4,
                                  padding='same'))
        model.add(Activation('tanh'))
        return model

    def build_dis(img_shape):
        model = Sequential()

        model.add(Conv1D(4, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(8, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(16, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(20, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Conv1D(40, kernel_size=3, activation='relu', strides=1, input_shape=img_shape,
                         padding='same'))

        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))
        return model

    def build_gan(gen, dis):
        model = Sequential()
        model.add(gen)
        model.add(dis)
        return model

    def train(iterations, batch_size, interval, Xtrain):

        Xtrain = np.asarray(Xtrain)
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            ids = np.random.randint(0, Xtrain.shape[0], batch_size)
            imgs = Xtrain[ids]

            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = gen_v.predict(z)

            dloss_real = dis_v.train_on_batch(imgs, real)
            dloss_fake = dis_v.train_on_batch(gen_imgs, fake)

            dloss, accuracy = 0.5 * np.add(dloss_real, dloss_fake)
            z = np.random.normal(0, 1, (batch_size, 100))
            gloss = gan_v.train_on_batch(z, real)

            if (iteration + 1) % interval == 0:
                losses.append((dloss, gloss))
                accuracies.append(100.0 * accuracy)
                iteration_checks.append(iteration + 1)
                print("%d [D loss: %f , acc: %.2f] [G loss: %f]" %
                      (iteration + 1, dloss, 100.0 * accuracy, gloss))

        return dis_v

    zdim = 100

    dis_v = build_dis((2000, 20))
    dis_v.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    gen_v = build_gen(zdim)
    dis_v.trainable = False
    gan_v = build_gan(gen_v, dis_v)
    gan_v.compile(loss='binary_crossentropy',
                  optimizer=Adam()
                  )

    losses = []
    accuracies = []
    iteration_checks = []

    dis_v = train(5000, 10, 100, XT)
    return dis_v


def build_GAN_A(FLAGS, XD_t, XT_t, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    gan_smiles = ganForDrug(XD_t)
    gan_protein = ganForTarget(XT_t)

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=32, input_length=1)(XDinput)
    encode_smiles = (gan_smiles.layers[-3])(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=FLAGS.charseqset_size + 1, output_dim=32, input_length=FLAGS.max_seq_len)(
        XTinput)
    encode_protein = (gan_protein.layers[-3])(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.25)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.25)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    predictions = Dense(1, kernel_initializer='normal')(FC2)
    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=[cindex_score])
    print(interactionModel.summary())
    return interactionModel


def build_GAN_B(FLAGS, XD_t, blosum_data, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    gan_smiles = ganForDrug(XD_t)
    gan_protein = ganForBlosumTarget(blosum_data)

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len, 20,), dtype='float32')

    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=32, input_length=1)(XDinput)
    encode_smiles = (gan_smiles.layers[-3])(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = (gan_protein.layers[-3])(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.25)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.25)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    predictions = Dense(1, kernel_initializer='normal')(FC2)
    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=[cindex_score])
    print(interactionModel.summary())
    return interactionModel


def build_GAN_C(FLAGS, XD_t, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    gan_smiles = ganForDrug(XD_t)

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len, 20,), dtype='float32')

    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=32, input_length=1)(XDinput)
    encode_smiles = (gan_smiles.layers[-3])(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.25)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.25)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    predictions = Dense(1, kernel_initializer='normal')(FC2)
    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=[cindex_score])
    print(interactionModel.summary())
    return interactionModel


def nfold_1_2_3_setting_sample(XD, XT, XD_t, XT_t, Y, label_row_inds, label_col_inds, runmethod, FLAGS, dataset):
    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS)
    foldinds = len(outer_train_sets)
    test_sets = []

    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    if FLAGS.model == 'A':
        bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv2(XD, XT,
                                                                                                               XD_t,
                                                                                                               XT_t, Y,
                                                                                                               label_row_inds,
                                                                                                               label_col_inds,
                                                                                                               runmethod,
                                                                                                               FLAGS,
                                                                                                               train_sets,
                                                                                                               val_sets)

        bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv2(XD, XT, XD_t, XT_t, Y,
                                                                                              label_row_inds,
                                                                                              label_col_inds,
                                                                                              runmethod, FLAGS,
                                                                                              train_sets,
                                                                                              test_sets)
    else:
        bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT,
                                                                                                              XD_t,
                                                                                                              XT_t, Y,
                                                                                                              label_row_inds,
                                                                                                              label_col_inds,
                                                                                                              runmethod,
                                                                                                              FLAGS,
                                                                                                              train_sets,
                                                                                                              val_sets)

        bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, XD_t, XT_t, Y,
                                                                                             label_row_inds,
                                                                                             label_col_inds,
                                                                                             runmethod, FLAGS,
                                                                                             train_sets,
                                                                                             test_sets)

    testperf = all_predictions[bestparamind]

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd


def general_nfold_cv(XD, XT, XD_t, XT_t, Y, label_row_inds, label_col_inds, runmethod, FLAGS, labeled_sets,
                     val_sets):
    paramset1 = FLAGS.num_windows
    paramset2 = FLAGS.smi_window_lengths
    paramset3 = FLAGS.seq_window_lengths
    epoch = FLAGS.num_epoch
    batchsz = FLAGS.batch_size

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    fpath = FLAGS.dataset_path
    CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                   "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                   "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                   "U": 19, "T": 20, "W": 21,
                   "V": 22, "Y": 23, "X": 24,
                   "Z": 25}
    index2char = {index: char for char, index in CHARPROTSET.items()}
    with open(fpath + "protein_feature_vecblsm.json") as f:
        pro2vec = json.load(f)

    if FLAGS.model == 'B':
        blosum_data = []
        new_XT = np.concatenate((XT, XT_t), axis=0)
        for i in range(10000):
            temp = []
            for j in range(len(new_XT[i])):
                try:
                    val = index2char[new_XT[i][j]]
                    temp.append(pro2vec[val])
                except:
                    arr = [0] * 20
                    temp.append(arr)
            blosum_data.append(temp)

    max_prot_len = FLAGS.max_seq_len

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = XD[trrows]
        XT_train = XT[trcols]
        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        pointer = 0

        train_generator = DataGenerator(train_drugs, train_prots, train_Y, pro2vec, batchsz, max_prot_len, index2char,
                                        shuffle=False)
        validation_generator = DataGenerator(val_drugs, val_prots, val_Y, pro2vec, batchsz, max_prot_len, index2char,
                                             shuffle=False)

        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]

            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]

                    if FLAGS.model == 'B':
                        gridmodel = runmethod(FLAGS, XD_t, blosum_data, param1value, param2value, param3value)
                    else:
                        gridmodel = runmethod(FLAGS, XD_t, param1value, param2value, param3value)

                    es = EarlyStopping(monitor='val_cindex_score', mode='max', verbose=1, patience=75,
                                       restore_best_weights=True)
                    gridres = gridmodel.fit_generator(generator=train_generator, validation_data=validation_generator,
                                                      epochs=epoch, callbacks=[es])

                    predicted_labels = gridmodel.predict_generator(validation_generator, verbose=0)

                    lst = gridres.history['val_cindex_score']
                    lst2 = gridres.history['val_loss']
                    rperf = max(lst)
                    index = lst.index(rperf)
                    loss = lst2[index]

                    rm2 = get_rm2(val_Y, predicted_labels)

                    predicted_labels = predicted_labels.tolist()
                    labels = []
                    for i in range(len(predicted_labels)):
                        labels.append(predicted_labels[i][0])

                    thresh = 7

                    temp = []
                    for i in range(len(val_Y)):
                        if (val_Y[i] > thresh):
                            temp.append(1)
                        else:
                            temp.append(0)

                    aupr = average_precision_score(temp, labels)

                    logging(
                        "P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI = %f, MSE = %f, aupr = %f , r2m = %f" %
                        (param1ind, param2ind, param3ind, foldind, rperf, loss, aupr, rm2), FLAGS)

                    df = pd.DataFrame(list(zip(labels, val_Y)),
                                      columns=['Predicted', 'Measured'])

                    x = sns.relplot(data=df, x="Predicted", y="Measured", s=20)
                    x.fig.set_size_inches(6.6, 5.5)

                    figname = "scatter_b" + str(param1ind) + "_e" + str(param2ind) + "_" + str(foldind) + "_" + str(
                        time.time())
                    plt.savefig("figures/" + figname + ".tiff", bbox_inches='tight', pad_inches=0.5, dpi=300,
                                pil_kwargs={"compression": "tiff_lzw"})
                    plt.close()

                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    all_predictions[pointer][foldind] = rperf
                    all_losses[pointer][foldind] = loss

                    pointer += 1
                    reset_keras()

    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):

                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf

                avgperf /= len(val_sets)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind, param3ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def general_nfold_cv2(XD, XT, XD_t, XT_t, Y, label_row_inds, label_col_inds, runmethod, FLAGS, labeled_sets,
                      val_sets):
    paramset1 = FLAGS.num_windows
    paramset2 = FLAGS.smi_window_lengths
    paramset3 = FLAGS.seq_window_lengths
    epoch = FLAGS.num_epoch
    batchsz = FLAGS.batch_size

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = XD[trrows]
        XT_train = XT[trcols]
        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        pointer = 0

        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]

            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]
                    gridmodel = runmethod(FLAGS, XD_t, XT_t, param1value, param2value, param3value)
                    es = EarlyStopping(monitor='val_cindex_score', mode='max', verbose=1, patience=75,
                                       restore_best_weights=True)
                    gridres = gridmodel.fit(([np.array(train_drugs), np.array(train_prots)]), np.array(train_Y),
                                            batch_size=batchsz, epochs=epoch,
                                            validation_data=(
                                                ([np.array(val_drugs), np.array(val_prots)]), np.array(val_Y)),
                                            shuffle=False, callbacks=[es])

                    predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots)])

                    lst = gridres.history['val_cindex_score']
                    lst2 = gridres.history['val_loss']
                    rperf = max(lst)
                    index = lst.index(rperf)
                    loss = lst2[index]

                    rm2 = get_rm2(val_Y, predicted_labels)

                    predicted_labels = predicted_labels.tolist()
                    labels = []
                    for i in range(len(predicted_labels)):
                        labels.append(predicted_labels[i][0])

                    thresh = 7

                    temp = []
                    for i in range(len(val_Y)):
                        if (val_Y[i] > thresh):
                            temp.append(1)
                        else:
                            temp.append(0)

                    aupr = average_precision_score(temp, labels)

                    logging(
                        "P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI = %f, MSE = %f, aupr = %f , r2m = %f" %
                        (param1ind, param2ind, param3ind, foldind, rperf, loss, aupr, rm2), FLAGS)

                    df = pd.DataFrame(list(zip(labels, val_Y)),
                                      columns=['Predicted', 'Measured'])

                    x = sns.relplot(data=df, x="Predicted", y="Measured", s=20)
                    x.fig.set_size_inches(6.6, 5.5)

                    figname = "scatter_b" + str(param1ind) + "_e" + str(param2ind) + "_" + str(foldind) + "_" + str(
                        time.time())
                    plt.savefig("figures/" + figname + ".tiff", bbox_inches='tight', pad_inches=0.5, dpi=300,
                                pil_kwargs={"compression": "tiff_lzw"})
                    plt.close()

                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    all_predictions[pointer][foldind] = rperf
                    all_losses[pointer][foldind] = loss

                    pointer += 1
                    reset_keras()

    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):

                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf

                avgperf /= len(val_sets)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind, param3ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf2.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g / f)  # select


def plotLoss(history, batchind, epochind, param3ind, foldind):
    figname = "b" + str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_" + str(foldind) + "_" + str(
        time.time())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/" + figname + ".png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()

    ## PLOT CINDEX
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/" + figname + "_acc.png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, deepmethod, foldcount=6):  # 5-fold cross validation + test
    dataset = DataSet(fpath=FLAGS.dataset_path,
                      setting_no=FLAGS.problem_type,
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y, XD_t, XT_t = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    XD_t = np.asarray(XD_t)
    XT_t = np.asarray(XT_t)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)  # basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, XD_t, XT_t, Y, label_row_inds,
                                                                    label_col_inds,
                                                                    deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    if FLAGS.model == 'B':
        deepmethod = build_GAN_B
    elif FLAGS.model == 'C':
        deepmethod = build_GAN_C
    else:
        deepmethod = build_GAN_A

    experiment(FLAGS, deepmethod)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression(FLAGS)
