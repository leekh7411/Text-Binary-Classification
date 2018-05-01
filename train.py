from model import TextBinaryClf
from config import get_config
from data_loader import KinQueryDataset,_batch_loader
import tensorflow as tf
import numpy as np
if __name__ == '__main__':

    config = get_config()

    if config.mode == 'train':

        model = TextBinaryClf(config)
        dataset = KinQueryDataset(config)

        one_batch_size = len(dataset) // config.batch
        if len(dataset) % config.batch != 0:
            one_batch_size += 1

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for epoch in range(config.epochs):

            s = np.random.permutation(dataset.labels.shape[0])
            dataset.vq1_train = dataset.vq1_train[s]
            dataset.vq2_train = dataset.vq2_train[s]
            dataset.labels = dataset.labels[s]

            avg_train_loss = 0
            avg_test_acc = 0

            for i, (data1, data2, labels) in enumerate(_batch_loader(dataset, config.batch)):

                train_loss = model.train(sess,data1,data2,labels)
                test_accuracy = model.predict_accuracy(sess,dataset.vq1_test,dataset.vq2_test,dataset.labels_test)

                print('Batch : ', i + 1, '/', one_batch_size, ', Batch Size:', one_batch_size,
                      'Train loss : ', float(train_loss),
                      "Valid score:", float(test_accuracy) * 100,
                      "Learning_rate:", (model.lr))

                avg_train_loss += train_loss
                avg_test_acc += test_accuracy

            print()
            print('Epoch:',epoch,'\ntrain_loss:', float(avg_train_loss / (one_batch_size)),'\nvalid_acc:',
                      float(avg_test_acc / (one_batch_size)) * 100)
            print('========================================================================================')

            model.lr_decay()
            model.save(sess,config.model_save_path)