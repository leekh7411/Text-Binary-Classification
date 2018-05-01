import tensorflow as tf
import numpy as np
import os
class TextBinaryClf():
    def __init__(self,config):
        self.sigle_model = None
        self.cfg = config
        self.fin_out_size = 1
        self.build_ensemble_model()
        self.build_loss_and_train_step()
        self.lr = self.cfg.lr

    def build_ensemble_model(self):
        self.x1 = tf.placeholder(tf.float32, [None, self.cfg.strmaxlen,self.cfg.w2v_size,1], name="x1-input")
        self.x2 = tf.placeholder(tf.float32, [None, self.cfg.strmaxlen,self.cfg.w2v_size,1], name="x2-input")
        self.y  = tf.placeholder(tf.float32, [None, self.fin_out_size], name="y-output")
        self.ensemble_models = []
        for i in range(3):
            self.ensemble_models.append(self.binary_input_cnn_rnn_model(self.x1, self.x2, char_size=251, embedding_size=8))

    def binary_input_cnn_rnn_model(self, input1, input2, char_size, embedding_size):
        #expand1, expand2 = self.embedding_layer(input1, input2, char_size, embedding_size)
        conv1 = self.cnn_layer(input1, filter=[4, 4], filter_size=16, index=1)
        conv2 = self.cnn_layer(input2, filter=[4, 4], filter_size=16, index=2)
        conv = tf.concat([conv1, conv2], axis=3)
        conv = self.cnn_layer(conv, [3, 3], 32, 0)
        flat = tf.layers.flatten(conv)
        dense = tf.layers.dense(flat, 256, activation=tf.nn.relu)
        model = self.rnn_layer(dense, 64, 1)
        return model

    def cnn_layer(self, input, filter, filter_size, index):
        with tf.name_scope("conv-%d" % index):
            input = tf.layers.conv2d(input, filter_size, filter, padding="SAME", activation=tf.nn.relu)
        with tf.name_scope("max_pool-%d" % index):
            input = tf.nn.relu(input)
            input = tf.layers.max_pooling2d(input, [2, 2], [2, 2], padding="SAME")
        input = tf.nn.l2_normalize(input, dim=1)
        return input

    def embedding_layer(self, input1, input2, char_size, embedding_size):
        # Embedding Input 1 and 2
        with tf.name_scope("embedding-1"):
            embedding_W1 = tf.Variable(
                tf.random_uniform([char_size, embedding_size], -1.0, 1.0),
                name="Embedding_W1"
            )

            embedding_W2 = tf.Variable(
                tf.random_uniform([char_size, embedding_size], -1.0, 1.0),
                name="Embedding_W2"
            )

            expand_input1 = tf.nn.embedding_lookup(embedding_W1, input1)
            expand_input2 = tf.nn.embedding_lookup(embedding_W2, input2)

            expand_input1 = tf.expand_dims(expand_input1, -1)
            expand_input2 = tf.expand_dims(expand_input2, -1)

        return expand_input1, expand_input2


    def rnn_layer(self, input, n_hidden, n_class):
        with tf.name_scope("rnn-layer"):
            input = tf.expand_dims(input, -1)
            # input = tf.reshape([-1, tf.shape(input)])
            cell = tf.nn.rnn_cell.LSTMCell(n_hidden, reuse=tf.AUTO_REUSE)
            outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]

        # outputs = tf.reshape([-1, tf.shape(outputs)])

        with tf.name_scope("Output-layer"):
            w = tf.Variable(tf.random_normal([n_hidden, n_class]))
            b = tf.Variable(tf.random_normal([n_class]))
            model = tf.nn.sigmoid(tf.matmul(outputs, w) + b)

        return model

    def build_loss_and_train_step(self):
        self.lr_tf = tf.placeholder(dtype=tf.float32,name="lr")
        with tf.name_scope("loss-optimizer"):
            # Binary Cross Entropy
            def binary_cross_entropy_loss(y_, output):
                # return tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output, 1e-10, 1.0))) - (1 - y_) * tf.log(tf.clip_by_value(1 - output, 1e-10, 1.0)))
                return tf.reduce_mean(-(y_ * tf.log(output)) - (1 - y_) * tf.log(1 - output))

            self.bce_loss = []
            for e_model in self.ensemble_models:
                self.bce_loss.append(binary_cross_entropy_loss(self.y, e_model))

            self.train_steps = []
            for loss in self.bce_loss:
                self.train_steps.append(tf.train.AdamOptimizer(learning_rate=self.lr_tf).minimize(loss))


    def save(self, sess,dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    def load(self, sess, dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    ''' for training '''

    def train(self,sess,data1,data2,labels):
        ensemble_loss = 0.
        for train, bce in zip(self.train_steps, self.bce_loss):
            _, loss = sess.run([train, bce],
                               feed_dict={
                                   self.x1: data1, self.x2: data2,
                                   self.y : labels,
                                   self.lr_tf: self.lr})
            ensemble_loss += loss
        ensemble_loss /= len(self.bce_loss)
        return ensemble_loss

    def lr_decay(self):
        self.lr = self.lr * self.cfg.decay

    ''' for predict input data '''

    def predict_accuracy(self,sess,data1,data2,labels):
        pred = self.predict(sess,data1,data2)
        is_correct = tf.equal(pred, labels)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        return sess.run(accuracy)

    def predict(self,sess,data1,data2):

        # For ensemble, predict each models
        def ensemble_pred(sess, model, data1, data2):
            pred = sess.run(model, feed_dict={self.x1: data1, self.x2: data2})
            pred_clipped = np.array(pred > self.cfg.threshold, dtype=np.float32)
            return pred_clipped


        preds = []
        for model in self.ensemble_models:
            preds.append(ensemble_pred(sess, model, data1, data2))

        # concat all predicted results([0.,1.,0.,1.,..],[1.,0.,1.,...],...) <- float data
        pred = tf.concat(preds, axis=1)

        # sum and mean all row data
        pred = tf.reduce_mean(pred, axis=1, keep_dims=True)

        # if five models result's is 0.8
        # --> [1,1,1,1,0] --> sum(4) --> mean(4/5) --> 0.8 --> threshold(0.5) --> 1
        # ensemble's result is '1'
        pred = np.array(sess.run(pred) > self.cfg.threshold, dtype=np.int)
        return pred

