from unittest.mock import DEFAULT


import datetime

FRAME_SIZE = 2048
HOP_LENGTH = 512
DURATION = 2.6
SAMPLE_RATE = 22050
MONO = True

#Default compile model
DEFAULT_LEARNING_RATE = 0.00025
DEFAULT_BATCH_SIZE = 3
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10

MODEL_PATH = "./model/"
MODEL_PATH_CLASSIQUE = "./model/classique_model.h5"
DEFAULT_MODEL_TUNING_TRIAL = 50

LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")