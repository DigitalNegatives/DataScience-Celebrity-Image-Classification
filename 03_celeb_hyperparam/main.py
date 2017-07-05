import tensorflow as tf
from utils.configuration import *
from utils.load_preprocessed import *
from utils.dataset import *
from utils.model import *
from utils.reporting import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from datetime import datetime


def start(params):

    tf.reset_default_graph()

    ### Get input parameters
    # tid = params['tid']
    xtrial = params['trial']
    print("len(xtrial):", len(xtrial))
    # print("xtrial[-1]:", xtrial[-1])
    # print("*** TID:", tid)
    activation_conv  = params['activation_conv']
    activation_fc = params['activation_fc']
    # orig_input_shape = params['orig_input_shape']
    # keep_prob = params['keep_prob']
    dropout_train = params['dropout_train']
    # learning_rate_bottleneck = params['learning_rate_bottleneck']
    learning_rate_bottleneck = params['learning_rate_bottleneck']
    learning_rate_label = params['learning_rate_label']
    n_filters = list(params['n_filters'])  #filter output sizes
    filter_sizes = list(params['filter_sizes'])
    #
    ds = params['ds']
    report_name = params['report_name']

    n_features = ds.X.shape[1]
    n_classes = ds.Y.shape[1]


    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_features], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, n_classes], name='y-input')
        z = tf.placeholder(tf.float32, [None, 2048], name='z-input')
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('input_reshape'):
        x_4d = tf.reshape(x, [-1, x_4d_shape[1], x_4d_shape[2], x_4d_shape[3]])
        tf.summary.image('input', x_4d, 10)

    # Create conlution layers
    h, Ws = convolutions(x_4d, n_filters, filter_sizes, filter_strides, act=activation_conv)
    h_shape = h.get_shape().as_list()
    # print("h.shape:", h.get_shape().as_list())
    # print("h[3]:", h.get_shape().as_list()[3])
    # print("Ws:", Ws)

    hidden1, pre_act_1 = nn_layer(h, h_shape[1]*h_shape[2]*h_shape[3], n_nodes, 'layer1', act=activation_fc)
    # print(hidden1)

    dropped = drop_layer(hidden1, keep_prob)
    hidden2, pre_act_2 = nn_layer(dropped, n_nodes, n_nodes, 'layer2')
    bottleneck, bottleneck_pre_act = nn_layer(hidden2, n_nodes, n_bottles, 'bottleneck')
    y, pre_act_3 = nn_layer(bottleneck, n_bottles, n_classes, 'layer3', act=tf.identity)

    # Cosine distance cost
    with tf.name_scope('cosine_distance'):
        bsum = tf.reduce_sum(tf.multiply(bottleneck_pre_act, z), 1)
        a = tf.sqrt(tf.reduce_sum(tf.square(bottleneck_pre_act), 1))
        b = tf.sqrt(tf.reduce_sum(tf.square(z), 1))
        ab = tf.multiply(a,b)
        cosine_distance = tf.reduce_mean(1 - (tf.div(bsum, ab)))
        cosine_distance_scaled = cosine_distance * beta

    loss_bottleneck = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(bottleneck, z)))
    optimizer_bottleneck = tf.train.AdamOptimizer(learning_rate_bottleneck).minimize(loss_bottleneck)

    # Label cost
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Optimizer
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate_label).minimize(cross_entropy + cosine_distance_scaled)

    # Accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Session creation
    merged = tf.summary.merge_all()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Bottleneck training
    avg_cost = 0
    if bottleneck_train:
        batch_size = 20

        for epoch_i in range(epochs_bottles):

            total_cost = []

            for batch_X, _, batch_Z in ds.train.next_batch(batch_size=batch_size):
                this_cost, _ = sess.run([loss_bottleneck, optimizer_bottleneck], feed_dict={x: batch_X - train_mean, z: batch_Z, keep_prob: dropout_train})
                total_cost = total_cost + this_cost

            if epoch_i%10 == 0:
                avg_cost = this_cost / (ds.X.shape[0] / batch_size)
                print(epoch_i, avg_cost)


    # Lable training
    batch_size = 20

    finalRepresentations = []
    acc_train_list = []
    for i in range(epochs_labels):

        if i%10 == 0:
            # Train accuracy report
            summary, acc_train, zbottleneck_pre_act = sess.run([merged, accuracy, bottleneck_pre_act], feed_dict={x:ds.train.images - train_mean,
                                                                                                                    y_:ds.train.labels,
                                                                                                                    keep_prob: dropout_test})
            acc_train_list.append(acc_train)
            if len(acc_train_list) > 4:
                acc_train_list.pop(0)
            finalRepresentations.append(zbottleneck_pre_act)
            train_writer.add_summary(summary, i)
            print('Train accuracy at step %s: %s' % (i, acc_train))
        else:
            # Train network
            for batch_X, batch_Y, batch_Z in ds.train.next_batch(batch_size=batch_size):
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_X,
                                                                       y_: batch_Y,
                                                                       z: batch_Z,
                                                                       keep_prob: dropout_train})
                train_writer.add_summary(summary, i)

    # Test set accuracy
    summary, acc_test, y = sess.run([merged, accuracy, y], feed_dict={x:ds.test.images - train_mean,
                                                                y_:ds.test.labels,
                                                                keep_prob: dropout_test})
    test_writer.add_summary(summary, i)
    print('Test Accuracy: %s' % (acc_test))

    # report(y, ds, n_classes)
    params_df = pd.DataFrame([params])
    acc_df = pd.DataFrame([{'acc_test':acc_test, 'acc_train':acc_train_list, 'avg_cost_bottle':avg_cost,'epochs_clsf':epochs_labels, 'epochs_bottles':epochs_bottles}])
    report_df = report(y, ds, n_classes)
    report_df = report_df.transpose()
    # report_df = report_df.append(params_df)
    # report_df = report_df.append(acc_df)


    print("\n+++++acc_df+++++: \n", acc_df)
    # print("report_df.type:", report_df.type)
    # print("report_df.to_html:", report_df.to_html(float_format='%.4f'))
    # with open(log_name, 'a') as f:
    # with open(report_name, 'a') as f:
        # f.write(report_df.to_html(float_format='%.4f'))

    with open(report_name, 'a') as f:
        f.write("<br>test:{}".format(xtrial.trials[-1]['tid']+1))
        f.write(params_df.to_html())
        f.write(report_df.to_html(float_format='%.4f'))
        f.write(acc_df.to_html())
        f.write("-------")


    train_writer.close()
    test_writer.close()
    sess.close()
    print("##########################\n\n\n\n")
    return {'loss': 1-acc_test, 'status':STATUS_OK, 'acc':acc_test}


############ MAIN ###############
if __name__ == "__main__":

    start_utc_time = datetime.now()
    # log_name = "results/results_{}.html".format(start_utc_time.strftime('%Y-%m-%d_%H%M'))
    log_csv_name = "results/results_{}.csv".format(start_utc_time.strftime('%Y-%m-%d_%H%M'))
    # print("log_name:", log_name)

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)
    report_name = "{}/results_{}.html".format(report_dir, start_utc_time.strftime('%Y-%m-%d_%H%M'))
    report_csv_name = "{}/results_{}.csv".format(report_dir , start_utc_time.strftime('%Y-%m-%d_%H%M'))
    print("report_name:", report_name)

    with open(report_name, 'a') as f:
        f.write("start time:{} <br>dataset:{}".format(str(start_utc_time), data_base_dir))
        # f.write("start time:" + str(start_utc_time))
        # f.write("<br>dataset:", data_base_dir)
        # f.write("<br>dataset:", data_base_dir)


    all_images_4d, all_labels, all_bottles = load_preprocessed_data(corpus_dir,
                                                bottleneck_dir)
    x_4d_shape = all_images_4d.shape
    all_images = np.ravel(all_images_4d).reshape(all_images_4d.shape[0],
            all_images_4d.shape[1] * all_images_4d.shape[2] * all_images_4d.shape[3])

    ds = Dataset(all_images, all_labels, all_bottles, split=split, one_hot=True, rnd_seed=seed)
    # n_samples = ds.X.shape[0]
    # n_features = ds.X.shape[1]
    # n_classes = ds.Y.shape[1]
    train_mean = np.mean(ds.train.images,0)

    df_train_class_sum, df_test_class_sum = get_class_distrutoins(ds)
    df_train_class_sum = df_train_class_sum.append(df_test_class_sum)

    with open(report_name, 'a') as f:
        f.write(df_train_class_sum.to_html(float_format='%.4f'))

    trials = Trials()
    best = fmin(start,
        algo=tpe.suggest,
        space={
            'trial':hp.choice('trial', [trials]),
            'report_name':hp.choice('report_name', [report_name]),
            # 'orig_input_shape':hp.choice('orig_input_shape', [x_4d_shape]),
            # 'train_mean':hp.choice('train_mean', [train_mean]),
            'ds':hp.choice('ds', [ds]),
            # 'n_filters':hp.choice('n_filters', [[32,32,16],[64,64,32]]),
            # 'n_filters':hp.choice('n_filters', n_filters),
            # 'filter_sizes':hp.choice('filter_sizes', [[5,5,3],[4,4,2],[3,3,2]]),
            # 'filter_sizes':hp.choice('filter_sizes', filter_sizes),
            # 'keep_prob':hp.choice('keep_prob', [0.5, 0.6, 0.7]),
            # 'dropout_train':hp.choice('dropout_train', [0.5, 0.6, 0.7]),
            # 'dropout_train':hp.choice('dropout_train', dropout_train),
            # 'activation_conv':hp.choice('activation_conv', [tf.nn.tanh, tf.nn.relu]),
            # 'activation_conv':hp.choice('activation_conv', [tf.nn.tanh]),
            # 'activation_fc':hp.choice('activation_fc', [tf.nn.tanh, tf.nn.relu]),
            # 'activation_fc':hp.choice('activation_fc', [tf.nn.sigmoid]),
            # 'learning_rate_bottleneck':hp.uniform('learning_rate_bottleneck', 0.0001, 0.0007),
            # 'learning_rate_bottleneck':hp.uniform('learning_rate_bottleneck', learning_rate_bottleneck),
            # 'learning_rate_label':hp.uniform('learning_rate_label', 0.0001, 0.0007)
            # 'learning_rate_label':hp.uniform('learning_rate_label', 0.0001, 0.0007)
            # 'learning_rate_label':hp.uniform('learning_rate_label', learning_rate_label)
            # # 'learning_rate_MSE':hp.normal('learning_rate_MSE', 0.00005, 2),
            # 'learning_rate':hp.normal('learning_rate', 0.00005, 2)
            'n_filters':config_n_filters,
            'filter_sizes':config_filter_sizes,
            'dropout_train':config_dropout_train,
            'activation_conv':config_activation_conv,
            'activation_fc':config_activation_fc,
            'learning_rate_bottleneck':config_learning_rate_bottleneck,
            'learning_rate_label':config_learning_rate_label
            },
        max_evals=1,
        trials=trials)
    print("Best:",best)
    stop_utc_time = datetime.now()
    with open(report_name, 'a') as f:
        f.write("<br>stop time:" + str(stop_utc_time))

    df_trials = pd.DataFrame(trials.trials)
    df_trials.to_csv(log_csv_name)
    with open('myTrial.html', 'a') as f:
        f.write(df_trials.to_html())

    print ("\n++++trails.trials+++++:", trials.trials)
    print ("\n++++trail.results+++++:", trials.results)
    print("Done")
