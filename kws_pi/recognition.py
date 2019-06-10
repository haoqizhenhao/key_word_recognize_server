#!/usr/bin/python3
#coding:utf-8
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pyaudio
from array import array
import collections
from collections import Counter
import sys
import signal
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 #

clip_stride_ms = 10 # 需要确保 RATE / clip_stride_ms 为整数
CHUNK_SIZE = int(RATE / clip_stride_ms) # 500
NUM_PADDING_CHUNKS = clip_stride_ms # 32
average_window_ms = 500
suppression_ms = 1500
detection_threshold = 0.8

average_window_samples = 7  #int(average_window_ms / clip_stride_ms)  # 15
suppression_samples = int(suppression_ms * RATE / 1000) # 240000

class KWS:
    def __init__(self, model_dir='model/ds_cnn1.pb'): ## model/ds_cnn.pb
        # load model
        self.sess = tf.InteractiveSession()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.load_pb(self.sess, pb_path=model_dir)
        self.labels = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        # self.labels_num = [0, 0, 1, ]
        print('Model restored.')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('labels_softmax:0')


    def record_and_recognize(self):
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
        signal.signal(signal.SIGINT, self.handle_int)
        suppression_flag = 0
        human_string_flags = ['none'] * average_window_samples
        human_string_index = 0
        score_flags = [0] * average_window_samples
        score_index = 0
        # print("* recording: ")
        stream.start_stream()
        data_save = b''
        while not leave and not got_10_result:
            chunk = stream.read(CHUNK_SIZE)
            if len(data_save) < 32000:
                data_save += chunk
            else:
                raw_data = array('h')
                raw_data.extend(array('h', data_save))
                data_save = data_save[CHUNK_SIZE*2:32000]
                data_save += chunk
                if suppression_flag == 0:
                    raw_data = np.array(raw_data,dtype=np.float32).reshape([16000,1])
                    raw_data = self.normalization(raw_data)
                    self.predictions = np.squeeze(self.sess.run(self.softmax_tensor, {'decoded_sample_data:0': raw_data}))
                    # Sort to show labels in order of confidence
                    top_1 = self.predictions.argsort()[-1]  # argsort()元素从小到大排列，提取其对应的index(索引)
                    # labels = self.load_labels(label_path)
                    human_string = self.labels[top_1]
                    score = self.predictions[top_1]
                    human_string_flags[human_string_index] = human_string
                    human_string_index += 1
                    human_string_index %= average_window_samples

                    score_flags[score_index] = score
                    score_index += 1
                    score_index %= average_window_samples

                    human_string_and_score_dict = self.counter(human_string_flags, score_flags)
                    human_string_big_score_tuple = sorted(human_string_and_score_dict.items(), key=lambda item:item[1])[0]
                    human_string = human_string_big_score_tuple[0]
                    score = human_string_big_score_tuple[1]

                    if score < detection_threshold or human_string == '_silence_' or human_string == '_unknown_' or human_string == 'none':
                        sys.stdout.write(str(0) + '\n')
                        # sys.stdout.flush()
                    else:
                        # sys.stdout.write(human_string + '(' + str(score) + ')')
                        sys.stdout.write(str(top_1)+'\n')
                        # sys.stdout.flush()
                        suppression_flag = 1
                        start = time.time()
                else:
                    if time.time()-start < suppression_ms / 1000:
                        chunk = stream.read(CHUNK_SIZE)
                        sys.stdout.write(str(0) + '\n')
                        # sys.stdout.flush()
                    else:
                        suppression_flag = 0
                sys.stdout.flush()
        sys.stdout.write('\n')
        stream.stop_stream()
        got_10_result = True
        leave = True
        stream.close()

    def handle_int(self, sig, chunk):
        global leave, got_10_result
        leave = True
        got_10_result = True

    def normalization(self, data):
        # 归一化数据到[-1,1]
        _range = np.max(abs(data))
        return data / _range

    def standardization(self, data):
        # 标准化
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

    def counter(self, human_string_arr, score_arr):
        # print(human_string_arr)
        top_num = 2
        string_top2 = Counter(human_string_arr).most_common(top_num)
        # print(string_top2)
        human_string_and_score_dict = {}
        if len(string_top2) == 1:
            human_string = string_top2[0][0]
            human_string_index = [j for j, x in enumerate(human_string_arr) if x == human_string]
            human_string_and_score_dict[human_string] = sum([score_arr[k] for k in human_string_index]) / len(
                human_string_arr)
        else:
            for i in range(top_num):
                human_string = string_top2[i][0]
                # print(human_string)
                human_string_index = [j for j, x in enumerate(human_string_arr) if x == human_string]
                human_string_and_score_dict[human_string] = sum([score_arr[k] for k in human_string_index]) / len(
                    human_string_arr)
        return human_string_and_score_dict

    def record_and_recognize1(self):
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
        signal.signal(signal.SIGINT, self.handle_int)
        suppression_flag = 0
        ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
        human_string_flags = ['none'] * average_window_samples
        human_string_index = 0
        score_flags = [0] * average_window_samples
        score_index = 0
        # ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
        # ring_buffer_index = 0
        # print("* recording: ")
        stream.start_stream()
        while not got_10_result and not leave:
            chunk = stream.read(CHUNK_SIZE)
            # print(chunk)
            ring_buffer.append(chunk)
            if suppression_flag == 0:
                if len(ring_buffer) < NUM_PADDING_CHUNKS:
                    continue
                # print(ring_buffer)
                data_save = b''
                for i in range(len(ring_buffer)):
                    data_save += ring_buffer[i]
                raw_data = array('h')
                raw_data.extend(array('h', data_save))
                # raw_data = np.interp(np.arange(0,16000), np.arange(0, RATE), raw_data)
                raw_data = np.array(raw_data,dtype=np.float32).reshape([16000,1])
                # print(len(raw_data))
                raw_data = self.normalization(raw_data)
                self.predictions = np.squeeze(self.sess.run(self.softmax_tensor, {'decoded_sample_data:0': raw_data}))
                # Sort to show labels in order of confidence
                top_1 = self.predictions.argsort()[-1]  # argsort()元素从小到大排列，提取其对应的index(索引)
                # labels = self.load_labels(label_path)

                human_string = self.labels[top_1]
                score = self.predictions[top_1]
                # print(human_string, str(score))

                human_string_flags[human_string_index] = human_string
                human_string_index += 1
                human_string_index %= average_window_samples

                score_flags[score_index] = score
                score_index += 1
                score_index %= average_window_samples

                human_string_and_score_dict = self.counter(human_string_flags, score_flags)
                human_string_big_score_tuple = sorted(human_string_and_score_dict.items(), key=lambda item:item[1])[0]
                human_string = human_string_big_score_tuple[0]
                score = human_string_big_score_tuple[1]

                if score < detection_threshold or human_string == '_silence_' or human_string == '_unknown_' or human_string == 'none':
                    sys.stdout.write(str(0) + '\n')
                else:
                    sys.stdout.write(str(top_1) + '\n')
                    # sys.stdout.write(human_string + '(' + str(score) + ')')
                    suppression_flag = 1
                    start = time.time()
            else:
                if time.time()-start < suppression_ms / 1000:
                    chunk = stream.read(CHUNK_SIZE)
                    sys.stdout.write(str(0) + '\n')
                    # sys.stdout.write('_')
                else:
                    suppression_flag = 0
            sys.stdout.flush()
        sys.stdout.write('\n')
        stream.stop_stream()
        # print("* done recording")
        got_10_result = True
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
    a = KWS()
    a.record_and_recognize1()
