import tensorflow as tf
from tensorflow.python.framework import graph_util
import sys
sys.path.append('../')


def freeze_graph(ckpt_name = '../model/20181229-191437/model-20181229-191437.ckpt-199207',
                 meta_name = '../model/20181229-191437/model-20181229-191437.meta',
                 pb_dir = '../model/pb/',
                 pb_name = 'frozen_model.pb',
                 last_layer_name = ['embeddings']):
    # create a session
    sess = tf.Session()
    # import best model
    saver = tf.train.import_meta_graph(meta_name) # graph
    saver.restore(sess, ckpt_name) # variables
    # get graph definition
    gd = sess.graph.as_graph_def()
    # fix batch norm nodes
    for node in gd.node:
      if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
          if 'moving_' in node.input[index]:
            node.input[index] = node.input[index] + '/read'
      elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    # generate protobuf
    converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, last_layer_name)
    tf.train.write_graph(converted_graph_def, pb_dir, pb_name, as_text=False)


if __name__ == '__main__':
    # freeze_graph(ckpt_name='../align/model/pnet/pnet-50',
    #              meta_name='../align/model/pnet/pnet-50.meta',
    #              pb_name='frozen_model_p.pb',
    #              last_layer_name=['landmark_pred'])

    # freeze_graph(ckpt_name='../align/model/rnet/rnet-50',
    #              meta_name='../align/model/rnet/rnet-50.meta',
    #              pb_name='frozen_model_r.pb',
    #              last_layer_name=['cls_fc','bbox_fc'])

    # freeze_graph(ckpt_name='../align/model/onet/onet-50',
    #              meta_name='../align/model/onet/onet-50.meta',
    #              pb_name='frozen_model_o.pb',
    #              last_layer_name=["landmark_fc"])

    freeze_graph(ckpt_name='../model/facenet_soft/20190102-162547/model-20190102-162547.ckpt-46',
                 meta_name='../model/facenet_soft/20190102-162547/model-20190102-162547.meta',
                 pb_dir='../model/pb/',
                 pb_name='frozen_model-soft-46.pb',
                 last_layer_name=['embeddings'])

