import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops,dtypes

def test1():
    keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords", 100)
    keywords_embedded = tf.feature_column.embedding_column(keywords, 16)
    columns = [keywords_embedded]
    features = {'keywords': tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
    'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
    'LSTM', 'Keras', 'RNN']])}
    input_layer = tf.keras.layers.DenseFeatures(columns)
    dense_tensor = input_layer(features)
    print(dense_tensor)

def test2():
    price = tf.feature_column.numeric_column('price')
    keywords_embedded = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_hash_bucket("keywords", 10000),
      dimension=16)
    columns = [price, keywords_embedded]
    print(tf.feature_column.make_parse_example_spec(columns))

    # feature_layer = tf.keras.layers.DenseFeatures(columns)
    # dataset = None
    # features = tf.io.parse_example(
    #   dataset, features=tf.feature_column.make_parse_example_spec(columns))
    # dense_tensor = feature_layer(features)
    # for units in [128, 64, 32]:
    #     dense_tensor = tf.keras.layers.Dense(units, activation='relu')(dense_tensor)
    # prediction = tf.keras.layers.Dense(1)(dense_tensor)
    # print(prediction)

def test3():
    keywords = tf.feature_column.categorical_column_with_hash_bucket("keywords", 8)
    inputs = {"keywords":tf.constant([['Tensorflow', 'Keras', 'RNN', 'LSTM',
    'CNN'], ['LSTM', 'CNN', 'Tensorflow', 'Keras', 'RNN'], ['CNN', 'Tensorflow',
    'LSTM', 'Keras', 'RNN']]), "ignore":tf.constant(["ignore"])}
    outputs = keywords._transform_feature(inputs)
    print(outputs)


def test4():
    train_x = np.random.random((500, 128))
    train_y = np.random.random((500, 10))

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    print(dataset)
    dataset = dataset.batch(100)
    print(dataset)
    # for i, d in enumerate(dataset):
    #     print("d:", i, d)
    dataset = dataset.repeat()
    print(dataset)
    # for i, d in enumerate(dataset):
    #     print("D:", i, d)

    inputs = tf.keras.Input(shape=(128,))
    layer1 = tf.keras.layers.Dense(384, activation='tanh', kernel_initializer='RandomUniform', name='layer1')(inputs)
    layer2 = tf.keras.layers.Dense(128, activation='elu')(layer1)
    layer3 = tf.keras.layers.Dense(64, activation="softplus", name='layer3')(layer2)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(layer3)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.05), loss = tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(dataset, epochs=50, steps_per_epoch=50)

def test5():
    #indices = ops.convert_n_to_tensor([[0,1],[0,2],[1,1]], dtype=dtypes.int64, name="indices")
    indices = ops.convert_to_tensor([[0,1],[0,2],[1,1]], dtype=dtypes.int64, name="indices")
    print(indices)
    print(type(indices.get_shape()), indices.get_shape(), indices.get_shape()[0])

    var = tf.Variable([1,2,3])
    print(var)

def test6():
    tf.compat.v1.disable_eager_execution()

    EPOCHS = 10
    BATCH_SIZE = 16
    features, labels = (np.array([np.random.sample((100,2))]), 
                    np.array([np.random.sample((100,1))]))
    #features, labels = (np.random.sample((100, 2)), np.random.sample((100, 1)))
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
    iter = dataset.make_one_shot_iterator()
    x, y = iter.get_next()
    net = tf.compat.v1.layers.dense(x, 8, activation=tf.tanh) # pass the first value
    net = tf.compat.v1.layers.dense(net, 8, activation=tf.tanh)
    prediction = tf.compat.v1.layers.dense(net, 1, activation=tf.tanh)
    loss = tf.compat.v1.losses.mean_squared_error(prediction, y) # pass the second value
    train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        '''
        for i in range(EPOCHS):
            xd, yd = sess.run([x, y])
            print(xd, yd)
            print(x, y)
        '''
        
        for i in range(EPOCHS):
            _, loss_value = sess.run([train_op, loss])
            print("Iter: {}, Loss: {:.4f}".format(i, loss_value))

def test7():
    tf.compat.v1.disable_eager_execution()

    EPOCHS = 10
    BATCH_SIZE = 2
    #x = np.random.sample((100,2))
    x = np.array([[1],[2],[3],[4]])
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(x)
    #dataset = dataset.repeat().batch(BATCH_SIZE)
    #dataset = dataset.shuffle(buffer_size = 4)
    dataset = dataset.batch(BATCH_SIZE).repeat()
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()
    with tf.compat.v1.Session() as sess:
        for i in range(EPOCHS):
            print(i, sess.run(el))

#test1()
#test2()
#test3()
#test4()
#test5()
#test6()
test7()
    
