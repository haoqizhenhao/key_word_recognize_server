import tensorflow as tf




# #
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import argparse
# import sys
# FLAGS = None
#
# def main():
#     mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#     x = tf.placeholder(tf.float32, shape=[None, 784])
#     w = tf.Variable(tf.random_normal([784, 10]))
#     b = tf.Variable(tf.zeros([10]))
#     y = tf.matmul(x, w) + b
#
#     y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#     train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
#     sess = tf.InteractiveSession()
#     tf.global_variables_initializer().run()
#
#     # train
#     for _ in range(1000):
#         batch_xs, batch_yx = mnist.train.next_batch(100)
#         sess.run(train_step, feed_dict={x: batch_xs, y_: batch_yx})
#
#     # test trained model
#     correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--data_dir', type=str, default='input_dir',
#         help='Directory for storing input data'
#     )
#     FLAGS, unparsed = parser.parse_known_args()
#     tf.app.run(main=main(), argv=[sys.argv[0]] + unparsed)