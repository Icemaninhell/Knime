import tensorflow as tf
import numpy as np
import datetime as dt
from tensorflow.keras.datasets import mnist

STORE_PATH = '/home/isra/PycharmProjects/CodingTensorFlowV2/StoredResults'

# Funciones

# Toma vectores aleatorios de un grupo
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]

# Definición del cálculo forward de la red
def nn_model(x_input, W1, b1, W2, b2):
    # flatten the input image from 28 x 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, W2), b2)
    return logits

# Definición de la función de pérdidas
def loss_fn(logits,labels):
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    return cross_entropy

# Fin Funciones

# Definición de red neuronal.
def nn_example():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Cargo de prueba un dataset de imágenes en 3D
    #(x_train_RGB, y_train_RGB), (x_test_RGB, y_test_RGB) = tf.keras.datasets.cifar10.load_data()

    # Variables de optimización
    epochs = 10
    batch_size = 100
    acc_limit = 0.98

    # Tamaño de capa intermedia
    int_size = 300

    # Normalización de la entrada
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # convert x_test to tensor to pass through model
    # (train data will be converted to tensors on the fly)
    x_test = tf.Variable(x_test)

    # Definición de la red neuronal.
    # Primera capa
    W1 = tf.Variable(tf.random.normal([784, int_size], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([int_size]), name='b1')
    # Segunda capa
    W2 = tf.Variable(tf.random.normal([int_size, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random.normal([10]), name='b2')

    #Definición del optimizador
    optimizer = tf.keras.optimizers.Adam()  #Puede tener un argumento de learning rate

    # Se crea un repositorio de escritura
    train_summary_writer = tf.summary.create_file_writer(STORE_PATH +
                                                         "/TensorFlow_Intro_Chapter_" +
                                                         f"{dt.datetime.now().strftime('%d%m%Y%H%M')}")

    # Cantidad de lotes de datos
    total_batch = int(len(y_train) / batch_size)

    for epoch in range(epochs):
        avg_loss = 0
        for i in range(total_batch):
            (batch_x, batch_y) = get_batch(x_train, y_train, batch_size=batch_size)
            # Creamos tensores
            batch_x = tf.Variable(batch_x)
            batch_y = tf.Variable(batch_y)
            #batch_y es un vector de números. Lo convertiremos en un one hot de la dimensión de los valores distintos
            batch_y = tf.one_hot(batch_y,10)
            #Novedad TF 2.x. Qué gradients queremos calcular. Las funciones en las que hay logits, se definen
            with tf.GradientTape() as tape:
                logits = nn_model(batch_x, W1, b1, W2, b2)
                loss = loss_fn(logits, batch_y)
            gradients = tape.gradient(loss, [W1, b1, W2, b2])
            optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
            avg_loss += loss / total_batch
        #Cálculo de la accuracy
        test_logits = nn_model(x_test, W1, b1, W2, b2)
        max_idxs = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
        print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy = {test_acc * 100: .3f} % ")

        #Visualización. Cuando queremos visualizar algo en TensorBoard, tenemos que crear un contexto.
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('accuracy', test_acc, step=epoch)

        if test_acc >= acc_limit:
            break

    print("\nTraining complete!")

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    nn_example()
