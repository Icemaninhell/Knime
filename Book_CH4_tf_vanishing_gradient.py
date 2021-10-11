import tensorflow as tf
from tensorflow import keras
import numpy as np

STORE_PATH = '/home/isra/PycharmProjects/CodingTensorFlowV2/StoredResults'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]


class Model(object):
    def __init__(self, activation, num_layers=6, hidden_size=10):
        # create the layers of the model
        self.num_layers = num_layers
        self.nn_model = tf.keras.Sequential()
        for i in range(num_layers):
            self.nn_model.add(tf.keras.layers.Dense(hidden_size, activation=activation, name=f'layer{i+1}'))
        # don't supply an activation for the final layer - the loss definition will
        # supply softmax activation. This defaults to a linear activation i.e. f(x) = x
        self.nn_model.add(tf.keras.layers.Dense(10, name='output_layer'))

    @tf.function()
    def forward(self, input_images):
        # flatten the input images
        input_images = tf.cast(input_images, tf.float32)
        input_images = tf.reshape(input_images, [-1, 28*28])
        input_images = input_images / 255.0
        logits = self.nn_model(input_images)
        return logits

    @staticmethod
    def loss(logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def log_gradients(self, gradients, train_writer, step):
        # the length of gradients will equal the length of the model trainable variables
        assert len(gradients) == len(self.nn_model.trainable_variables)
        # there are 2 trainable variables per layer - weight and bias
        for i in range(len(gradients)):
            if 'kernel' in self.nn_model.trainable_variables[i].name:
                with train_writer.as_default():
                    tf.summary.scalar(f"mean_{int((i - 1) / 2)}", tf.reduce_mean(tf.abs(gradients[i])), step=step)
                    tf.summary.histogram(f"histogram_{int((i - 1) / 2)}", gradients[i], step=step)
                    tf.summary.histogram(f"hist_weights_{int((i - 1) / 2)}", self.nn_model.trainable_variables[i],
                                         step=step)

    def plot_computational_graph(self, train_writer, x_batch):
        tf.summary.trace_on(graph=True)
        self.forward(x_batch)
        with train_writer.as_default():
            tf.summary.trace_export(name="graph", step=0)

def run_training(model: Model, sub_folder: str, iterations: int = 2500, batch_size: int = 32, log_freq: int = 200):
    train_writer = tf.summary.create_file_writer(STORE_PATH + "/" + sub_folder)
    model.plot_computational_graph(train_writer, x_train[:batch_size, :, :])
    # setup the optimizer
    optimizer = tf.keras.optimizers.Adam()
    for i in range(iterations):
        image_batch, label_batch = get_batch(x_train, y_train, batch_size)
        label_batch = tf.cast(tf.Variable(label_batch), tf.int32)
        with tf.GradientTape() as tape:
            logits = model.forward(image_batch)
            loss = model.loss(logits, label_batch)
        gradients = tape.gradient(loss, model.nn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.nn_model.trainable_variables))
        if i % log_freq == 0:
            max_idxs = tf.argmax(logits, axis=1)
            acc = np.sum(max_idxs.numpy() == label_batch.numpy()) / len(label_batch.numpy())
            print(f"Iter: {i}, loss={loss:.3f}, accuracy={acc * 100:.3f}%")
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=i)
                tf.summary.scalar('accuracy', acc, step=i)
            # log the gradients
            model.log_gradients(gradients, train_writer, i)


if __name__ == "__main__":
    scenarios = ["sigmoid", "relu", "leaky_relu"]
    act_funcs = [tf.sigmoid, tf.nn.relu, tf.nn.leaky_relu]
    assert len(scenarios) == len(act_funcs)
    # collect the training data
    for i in range(len(scenarios)):
        print(f"Running scenario: {scenarios[i]}")
        model = Model(act_funcs[i], 6, 10)
        run_training(model, scenarios[i])