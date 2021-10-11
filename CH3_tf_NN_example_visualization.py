import tensorflow as tf
import numpy as np
import datetime as dt
from tensorflow.keras.datasets import mnist
import os

STORE_PATH = '/home/isra/PycharmProjects/CodingTensorFlowV2/StoredResults/CH3_Visualization'

# Funciones

# Toma vectores aleatorios de un grupo
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]

# Definición del cálculo forward de la red
@tf.function
#Modificada respecto al ejemplo anterior
#Se mete dentro la función de cálculo de la entropía cruzada (loss_fn)
def nn_model(x_input, labels, W1, b1, W2, b2):
    # flatten the input image from 28 x 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    with tf.name_scope("Hidden") as scope:
        hidden_logits = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
        hidden_out = tf.nn.leaky_relu(hidden_logits)
    with tf.name_scope("Output") as scope:
        logits = tf.add(tf.matmul(hidden_out, W2), b2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                           logits=logits))
    return logits, hidden_logits, hidden_out, cross_entropy

# Fin Funciones

# Definición de red neuronal.
def nn_example():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Cargo de prueba un dataset de imágenes en 3D
    #(x_train_RGB, y_train_RGB), (x_test_RGB, y_test_RGB) = tf.keras.datasets.cifar10.load_data()

    # Variables de optimización
    epochs = 30
    batch_size = 100
    acc_limit = 0.975
    int_size = 300          # Tamaño de capa intermedia

    # Normalización de la entrada
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # convert x_test to tensor to pass through model
    # (train data will be converted to tensors on the fly)
    x_test = tf.Variable(x_test)
    y_test = tf.Variable(y_test)

    # Definición de la red neuronal.
    # Primera capa
    W1 = tf.Variable(tf.random.normal([784, int_size], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([int_size]), name='b1')
    # Segunda capa
    W2 = tf.Variable(tf.random.normal([int_size, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random.normal([10]), name='b2')

    #Definición del optimizador
    optimizer = tf.keras.optimizers.Adam()  #Puede tener un argumento de learning rate

    #Visualización
    #Foto del modelo. Llamamos a la función que contiene el modelo aunque no sirva para nada
    out_file = STORE_PATH + f"/TensorFlow_Visualization_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
    train_summary_writer = tf.summary.create_file_writer(out_file)
    batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
    batch_y = tf.one_hot(batch_y, 10)
    tf.summary.trace_on(graph=True)
    logits, _, _, _ = nn_model(batch_x, batch_y, W1, b1, W2, b2)
    with train_summary_writer.as_default():
        tf.summary.trace_export(name="graph", step=0)

    # Cantidad de lotes de datos
    total_batch = int(len(y_train) / batch_size)

    #Bucle de entrenamiento
    for epoch in range(epochs):
        avg_loss = 0
        for i in range(total_batch):
            (batch_x, batch_y) = get_batch(x_train, y_train, batch_size=batch_size)
            # Creamos tensores
            #batch_x = tf.Variable(batch_x)
            #batch_y = tf.Variable(batch_y)
                #Lo anterior puede no ser necesario
            #batch_y es un vector de números. Lo convertiremos en un one hot de la dimensión de los valores distintos
            batch_y = tf.one_hot(batch_y,10)
            #Novedad TF 2.x. Qué gradients queremos calcular. Las funciones en las que hay logits, se definen
            with tf.GradientTape() as tape:
                logits, hidden_logits, hidden_out, loss = nn_model(batch_x, batch_y, W1, b1, W2, b2)
            gradients = tape.gradient(loss, [W1, b1, W2, b2])
            optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
            avg_loss += loss / total_batch
        #Cálculo de la accuracy
        test_logits, _, _, _ = nn_model(x_test, tf.one_hot(y_test, 10), W1, b1, W2, b2)
        max_idxs = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idxs.numpy() == y_test.numpy()) / len(y_test.numpy())
        print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy = {test_acc * 100: .3f} % ")

        #Clasificación de las imágenes
        correct_inputs = tf.boolean_mask(x_test, max_idxs.numpy() == y_test.numpy())
        incorrect_inputs = tf.boolean_mask(x_test, tf.logical_not(max_idxs.numpy() == y_test.numpy()))

        #Visualización. Cuando queremos visualizar algo en TensorBoard, tenemos que crear un contexto.
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('accuracy', test_acc, step=epoch)
            tf.summary.histogram('hidden_out', hidden_out, step=epoch)
            tf.summary.histogram('hidden_logits', hidden_logits, step=epoch)
            tf.summary.image('correct_images {}'.format(epoch), tf.reshape(correct_inputs, (-1, 28, 28, 1)), max_outputs=10,
                             step=epoch)
            tf.summary.image('incorrect_images {}'.format(epoch), tf.reshape(incorrect_inputs, (-1, 28, 28, 1)), max_outputs=10,
                             step=epoch)

        if test_acc >= acc_limit:
            break

    print("\nTraining complete!")
    print("\nWill execute TensorBoard...")
    os.system("tensorboard --logdir={}".format(out_file))

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    nn_example()
