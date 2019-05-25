import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np


class KWS:
    def __init__(self, model_dir='model/CNN_L.pb'):
        # load model
        self.sess = tf.InteractiveSession()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.load_pb(self.sess, pb_path=model_dir)
        print('Model restored.')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('labels_softmax:0')


    def rec(self, wav = 'wav/test1.wav', label_path='label/labels.txt',
                 num_top_predictions=3):
        # 识别函数
        with open(wav, 'rb') as wav_file:
            wav_data = wav_file.read()
        self.predictions = np.squeeze(self.sess.run(self.softmax_tensor, {'wav_data:0': wav_data}))
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


    def load_pb(self, sess, pb_path):
        # pb模型导入
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图


    def load_labels(self, filename):
        """Read in labels, one label per line."""
        return [line.rstrip() for line in tf.gfile.GFile(filename)]


# if __name__ == '__main__':
#     print(tf.__version__)
#     a = KWS()
#     print(a.rec())

