import numpy as np
from deepspeaker.audio_ds import read_mfcc
from deepspeaker.batcher import sample_from_mfcc
from deepspeaker.constants import SAMPLE_RATE, NUM_FRAMES, WIN_LENGTH
from deepspeaker.conv_models import DeepSpeakerModel
import tensorflow as tf

def build_model(ckpt_path):
    model = DeepSpeakerModel()
    model.m.load_weights(ckpt_path, by_name=True)
    return model

def predict_embedding(model, audio, sr=SAMPLE_RATE, win_length=WIN_LENGTH, cuda=True):
    mfcc = sample_from_mfcc(read_mfcc(audio, sr, win_length), NUM_FRAMES)
    # Call the model to get the embeddings of shape (1, 512) for each file.
    gpus = tf.config.experimental.list_physical_devices('GPU') if cuda else 0
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)
        with tf.device('/device:GPU:0'):
            embedding = model.m.predict(np.expand_dims(mfcc, axis=0)) # Female
    else:
        with tf.device('device:cpu:0'):
            embedding = model.m.predict(np.expand_dims(mfcc, axis=0)) # Female
    return embedding
