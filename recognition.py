#coding:utf-8
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pyaudio
import wave
from struct import pack
import json
from array import array
import collections
import sys
import signal
from threading import Thread
import queue, time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 800
CHUNK_DURATION_MS = 100
NUM_PADDING_CHUNKS = int(RATE / CHUNK_SIZE)
NUM_WINDOW_CHUNKS = 13


def record_to_file(path, data, sample_width):
    "Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def handle_int(sig, chunk):
    global leave, got_10_result
    leave = True
    got_10_result = True


def normalization(data):
    # 归一化数据到[-1,1]
    _range = np.max(abs(data))
    return data / _range


def standardization(data):
    # 标准化
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

class KWS:
    def __init__(self, model_dir='model\dnn2.pb'): #model/CNN_L.pb  model/dnn.pb model\Pretrained_models\DS_CNN/DS_CNN_L.pb
        # load model
        self.sess = tf.InteractiveSession()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.load_pb(self.sess, pb_path=model_dir)
        print('Model restored.')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('labels_softmax:0')
        self.sample_rate = 16000

    def recognize_file(self, wav='wav/test.wav', label_path='label/labels.txt',
                       num_top_predictions=3):
        # recognize
        with open(wav, 'rb') as wav_file:
            wav_data = wav_file.read()
        print(wav_data)
        self.predictions = np.squeeze(self.sess.run(self.softmax_tensor, {'decoded_sample_data:0': wav_data})) # decoded_sample_data:0  wav_data:0
        # Sort to show labels in order of confidence
        top_k = self.predictions.argsort()[-num_top_predictions:]  # argsort()元素从小到大排列，提取其对应的index(索引)
        labels = self.load_labels(label_path)
        result = []
        for node_id in top_k:
            human_string = labels[node_id]
            score = self.predictions[node_id]
            result.append('%s (score = %.5f)' % (human_string, score))
            print('%s (score = %.5f)' % (human_string, score))
        print(result)
        return '\n'.join(result)

    def recognize_realtime(self, wav_stream, label_path='label/labels.txt', num_top_predictions=3):
        wav_stream = wav_stream.read()
        print(wav_stream)
        print(int.from_bytes(wav_stream, byteorder='big'))
        self.predictions = np.squeeze(self.sess.run(self.softmax_tensor, {'wav_data:0': wav_stream}))
        # Sort to show labels in order of confidence
        top_k = self.predictions.argsort()[-num_top_predictions:]  # argsort()元素从小到大排列，提取其对应的index(索引)
        labels = self.load_labels(label_path)
        result = []
        for node_id in top_k:
            human_string = labels[node_id]
            score = self.predictions[node_id]
            result.append('%s (score = %.5f)' % (human_string, score))
            print('%s (score = %.5f)' % (human_string, score))
        # print(result)
        return '\n'.join(result)


    def record(self, label_path='label/labels.txt'):
        flag = 0
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=RATE,
                         input=True,
                         start=False,
                         # input_device_index=2,
                         frames_per_buffer=CHUNK_SIZE)
        leave = False
        got_10_result = False
        signal.signal(signal.SIGINT, handle_int)
        # print('raw_data', raw_data)
        while not leave:
            ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
            ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
            ring_buffer_index = 0
            print("* recording: ")
            stream.start_stream()
            while not got_10_result and not leave:
                chunk = stream.read(CHUNK_SIZE)
                ring_buffer.append(chunk)
                if len(ring_buffer) < NUM_PADDING_CHUNKS:
                    continue
                # print(ring_buffer)
                data_save = b''
                for i in range(len(ring_buffer)):
                    data_save += ring_buffer[i]
                raw_data = array('h')
                raw_data.extend(array('h', data_save))
                raw_data = np.array(raw_data,dtype=np.float32).reshape([16000,1])
                raw_data = normalization(raw_data)

                # print(raw_data)
                # raw_data = tf.reshape(raw_data,shape=[16000,1])
                # record_to_file('aa.wav', raw_data, 2)
                # with open('aa.wav', 'rb') as wav_file:
                #     wav_data = wav_file.read()
                # raw_data = np.frombuffer(chunk, dtype=np.uint8)
                # print(raw_data)
                # print(len(raw_data))
                self.predictions = np.squeeze(self.sess.run(self.softmax_tensor, {'decoded_sample_data:0': raw_data}))
                # Sort to show labels in order of confidence
                top_1 = self.predictions.argsort()[-1]  # argsort()元素从小到大排列，提取其对应的index(索引)
                labels = self.load_labels(label_path)
                human_string = labels[top_1]

                score = self.predictions[top_1]
                # print(human_string, str(score))
                if human_string == '_silence_' or human_string == '_unknown_' or score <= 0.7: # or score <= 0.4
                    active = False
                else:
                    active = True
                    flag += 1
                ring_buffer_flags[ring_buffer_index] = 1 if active else 0
                ring_buffer_index += 1
                ring_buffer_index %= NUM_WINDOW_CHUNKS
                num_voiced = sum(ring_buffer_flags)
                # print(ring_buffer_flags)
                if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                    sys.stdout.write(human_string)
                else:
                    sys.stdout.write('_')

                sys.stdout.flush()
                if flag >= 1000:
                    got_10_result = True
            sys.stdout.write('\n')
            stream.stop_stream()
            print("* done recording")
            got_10_result = False
            leave = True
        stream.close()

    def load_pb(self, sess, pb_path):
        # pb模型导入
        with gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

    def load_labels(self, filename):
        """Read in labels, one label per line."""
        return [line.rstrip() for line in tf.gfile.GFile(filename)]


if __name__ == '__main__':
    print(tf.__version__)
    wav = 'wav/test.wav'
    a = KWS()
    # print(a.recognize_file())
    # with open(wav, 'rb') as wav_file:
    #     wav_data = wav_file.read()
    # a.recognize_realtime(wav_stream=wav_data)
    # while True:
    a.record()
