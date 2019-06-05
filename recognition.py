#!/usr/bin/python3
#coding:utf-8
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pyaudio
import wave
from struct import pack
from array import array
import collections
from collections import Counter
import sys
import signal
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
clip_stride_ms = 32 # 需要确保 RATE / clip_stride_ms 为整数
CHUNK_SIZE = int(RATE / clip_stride_ms) # 500
NUM_PADDING_CHUNKS = clip_stride_ms # 32
NUM_WINDOW_CHUNKS = 13
average_window_ms = 500
suppression_ms = 1500
detection_threshold = 0.9

average_window_samples = int(average_window_ms / clip_stride_ms)+2  # 15
suppression_samples = int(suppression_ms * RATE / 1000) # 240000


class KWS:
    def __init__(self, model_dir='model\ds_cnn.pb'): #model/CNN_L.pb  model/dnn.pb model\Pretrained_models\DS_CNN/DS_CNN_L.pb
        # load model
        self.sess = tf.InteractiveSession()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.load_pb(self.sess, pb_path=model_dir)
        print('Model restored.')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('labels_softmax:0')

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

    def record_to_file(self, path, data, sample_width):
        "Records from the microphone and outputs the resulting data to 'path'"
        # sample_width, data = record()
        data = pack('<' + ('h' * len(data)), *data)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()

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
        signal.signal(signal.SIGINT, self.handle_int)
        # print('raw_data', raw_data)
        while not leave:
            suppression_flag = 0
            ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
            human_string_flags = ['none'] * average_window_samples
            human_string_index = 0
            score_flags = [0] * average_window_samples
            score_index = 0
            # ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
            # ring_buffer_index = 0
            print("* recording: ")
            stream.start_stream()
            while not got_10_result and not leave:

                chunk = stream.read(CHUNK_SIZE)
                # print(chunk)
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
                raw_data = self.normalization(raw_data)
                if suppression_flag == 0:
                    self.predictions = np.squeeze(self.sess.run(self.softmax_tensor, {'decoded_sample_data:0': raw_data}))
                    # Sort to show labels in order of confidence
                    top_1 = self.predictions.argsort()[-1]  # argsort()元素从小到大排列，提取其对应的index(索引)
                    labels = self.load_labels(label_path)
                    human_string = labels[top_1]
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
                        sys.stdout.write('_')
                    else:
                        sys.stdout.write(human_string + '(' + str(score) + ')')
                        # sys.stdout.write(human_string)
                        # flag += 1
                        suppression_flag = 1
                        start = time.time()
                else:
                    if time.time()-start < suppression_ms / 1000:
                        chunk = stream.read(CHUNK_SIZE)
                        sys.stdout.write('_')
                    else:
                        suppression_flag = 0
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
    a = KWS()
    a.record()
