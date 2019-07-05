import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
from create_record_files import get_example_nums
import PARAMS as Param
from models import dense_net as net
import dataset_factory.dataset_factory as datasets
from tensorflow.python.ops import control_flow_ops

PARAMS = Param.Params()
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_CLASS = PARAMS.params['model']['classNum']
WEIGHT_DECAY = 1e-5
KEEP_PROB = 0.6
BASE_LR1 = 1e-3
BASE_LR2 = 5e-4
IMAGE_HEIGHT = PARAMS.params['model']['height']
IMAGE_WIDTH = PARAMS.params['model']['width']
IMAGE_DEPTH = PARAMS.params['model']['depth']
TENSORBOARD_PATH = PARAMS.params['path']['train_tensorBoardPath']

#读入文件路径
train_dir = os.path.dirname(PARAMS.params['path']['train_rex'])
print(train_dir)
train_num = get_example_nums(train_dir)
print("train num = {0}".format(train_num))
val_dir = os.path.dirname(PARAMS.params['path']['val_rex'])
val_num = get_example_nums(val_dir)
print("validation num = {0}".format(val_num))



def train():
    # print_tensors_in_checkpoint_file(MODEL_PATH, None, False, True)
    #定义数据集
    training_dataset = datasets.get_dataset('iwatch', train_dir,
                                            'train', batch_size=BATCH_SIZE)
    # train_dataset 用epochs控制循环
    validation_dataset = datasets.get_dataset('iwatch', val_dir,
                                            'test', batch_size=BATCH_SIZE)

    train_iterator = training_dataset.make_initializable_iterator()
    # make_initializable_iterator 每个epoch都需要初始化
    val_iterator = validation_dataset.make_initializable_iterator()
    # make_one_shot_iterator不需要初始化，根据需要不停循环
    train_images, train_labels = train_iterator.get_next()
    val_images, val_labels = val_iterator.get_next()

    # 定义输入
    with tf.name_scope("inputs"):
        is_training = tf.placeholder(tf.bool)
        images = tf.placeholder(
            dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], name='inputs')
        labels = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS], name='label')

    with tf.name_scope('nets'):
        logits, endPoints = net.inference(inputs=images, num_classes=NUM_CLASS,
                                         is_training=is_training, dropout_keep_prob = KEEP_PROB)

    with tf.name_scope("loss_function"):
        tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        loss_entro = tf.losses.get_losses()
        loss_entro_mean = tf.reduce_mean(loss_entro)
        tf.summary.scalar("cross_entropy", loss_entro_mean)
    # prediction


    with tf.name_scope("train_op"):
        # fc8_optimizer = tf.train.GradientDescentOptimizer(BASE_LR1)  不能定义太多优化器，内存会爆掉
        global_step = tf.train.create_global_step()
        learning_rate = tf.train.exponential_decay(
            PARAMS.params['model']['baseLR'],
            global_step,
            4*train_num / BATCH_SIZE,
            PARAMS.params['model']['decayLR'])

        optimizer = tf.train.AdamOptimizer(learning_rate)

        full_train_op = slim.learning.create_train_op(loss_entro_mean,
                                                      optimizer,
                                                      global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            print("BN parameters: ", update_ops)
            updates = tf.group(*update_ops)
            full_train_op = control_flow_ops.with_dependencies([updates], full_train_op)
        for v in tf.all_variables():
            print(v.name)
            if 'batch_normalization' in v.name:
                tf.summary.histogram(v.name, v)

        # 冻结fc7 以前的层
        # fc8_train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=fc8_variables)
        # 全部训练

    with tf.name_scope("accuracy"):
        prediction = tf.argmax(logits, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("input"):
        image_shape_input = tf.reshape(images, [-1, 224, 224, 1])
        tf.summary.image('input_image', image_shape_input, 10)

    with tf.name_scope("feature_map"):
        feture_maps = endPoints["layer1"]
        # 取其中的第一张图像
        feture_maps = tf.slice(feture_maps, (0, 0, 0, 0), (1, -1, -1, -1))
        ix = iy = 56
        ix += 4
        iy += 4
        #类似subplot的方法图像显示为4x4
        cy = cx = 4
        #做padding方便将所有图分开
        feture_maps = tf.image.resize_image_with_crop_or_pad(feture_maps, iy, ix)
        # reshape 成 56x56x4x4
        feture_maps = tf.reshape(feture_maps, (iy, ix, cy, cx))
        #交换维度
        feture_maps = tf.transpose(feture_maps, (2, 0, 3, 1))  # cy,iy,cx
        #将所有通道的图片组成一张图片显示
        feture_maps = tf.reshape(feture_maps, (1, cy * iy, cx * ix, 1))

        tf.summary.image("layer1_feature_map", feture_maps, 1)

    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    saver = tf.train.Saver()
    summary_merge = tf.summary.merge_all()

    # 结束当前的计算图，使之成为只读
    tf.get_default_graph().finalize()
    with tf.Session() as sess:
        # 先初始化网络
        sess.run(init_op)
        tensorboard_writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)
        max_acc = 0.70
        for epoch in range(NUM_EPOCHS):
            print('Starting epoch %d / %d' % (epoch, NUM_EPOCHS))
            sess.run(train_iterator.initializer)
            while True:
                try:
                    train_batch_images, train_batch_labels \
                        = sess.run([train_images, train_labels])
                    _,summary, train_loss, train_acc = sess.run([full_train_op, summary_merge,loss_entro_mean, accuracy],
                                                         feed_dict={is_training: True,
                                                                images: train_batch_images,
                                                                labels: train_batch_labels})
                    step = sess.run(global_step)
                    tensorboard_writer.add_summary(summary,step)
                    print("epoch = {0} step = {3}: train-loss = {1}, batch-acc = {2}".format(epoch, train_loss,train_acc,step))
                except tf.errors.OutOfRangeError:
                    break
            sess.run(val_iterator.initializer)
            num_correct = 0
            while True:
                try:
                    val_batch_images, val_batch_labels \
                        = sess.run([val_images, val_labels])

                    correct_pred = sess.run(correct_prediction, feed_dict={
                                                        is_training: False,
                                                        images: val_batch_images,
                                                        labels: val_batch_labels})

                    num_correct += correct_pred.sum()
                except tf.errors.OutOfRangeError:
                    break
            val_acc = float(num_correct) / val_num
            if val_acc > max_acc or epoch > NUM_EPOCHS-2:
                max_acc = val_acc
                best_models = os.path.join(PARAMS.params['path']['model_path'],
                                           'model_epoch{}_{:.4f}.ckpt'.format(epoch, val_acc))
                saver.save(sess,best_models)
            print("epoch = {0}, val-acc = {1}".format(epoch, val_acc))
        tensorboard_writer.close()

if __name__ == '__main__':
    train()
