import tensorflow as tf
import numpy as np
import ml_functions as mli
import datetime as dt
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mplt
import sys

# Check TF uses GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#Management of parameters in execution

n = len(sys.argv)
print(f"\033[1;32m\nExecuting: {sys.argv[0]}")
print("Total arguments passed:", n-1)

assert n <= 2

if n == 1:
    l_space = int(3)
if n == 2:
    l_space = int(sys.argv[1])

assert l_space <= 3, "Dimension should be 2 or 3"

print(f"Latent space dimension: {l_space}\033[0;0m")

# Dataset download

STORE_PATH = f"/home/isra/PycharmProjects/CodingTensorFlowV2/StoredResults/KNIME/3D_Midpoint_DM/{dt.datetime.now().strftime('%d%m%Y%H%M')}"

mnist = fetch_openml('mnist_784', version=1, as_frame=True)

df_data = mnist.data.iloc[:]
df_data = df_data.assign(digit_class=mnist.target.iloc[:].astype(int))
np_data = df_data.to_numpy()
np_data_images = np_data[:, 0:784]
np_data_images = np.reshape(np_data_images, (70000, 28, 28))
np_data_labels = np_data[:, 784]

#Partitioning, Test, Train, Validation

x_train,x_validation, y_train, y_validation = train_test_split(np_data_images, np_data_labels, test_size=0.25, random_state=42)
x_train,x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

# Datasets capturados

# Función de entrenamiento
def run_training_enc(model_inst: mli.ModelGr, sub_folder: str, iterations: int = 2500,
                 batch_size: int = 32, log_freq: int = 200, lim_accuracy: float = 0.99,
                 lim_loss: float = 0.05, graph_name: str = None):

    # Directorio para almacenar datos
    train_writer = tf.summary.create_file_writer(STORE_PATH + "/" + sub_folder)
    # Guardamos el grafo
    model_inst.plot_computational_graph(train_writer, x_train[:batch_size, :, :], graph_name)
    # Selección del optimizador
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)  # Puede tener un argumento de learning rate
    # Vamos a la iteración de entrenamiento
    acc = 0
    for j in range(iterations):
        # Tomamos un batch
        image_batch, label_batch = mli.get_batch(x_train, y_train, batch_size)
        # image_batch = tf.Variable(image_batch) (Está en el libro pero no funciona)
        label_batch = tf.cast(tf.Variable(label_batch), tf.int32)
        # Calculamos los logits y la perdida
        with tf.GradientTape() as tape:
            logits = model_inst.forward(image_batch)
            loss = model_inst.loss(logits, label_batch)
        gradients = tape.gradient(loss, model_inst.nn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_inst.nn_model.trainable_variables))
        # Zona de Logs
        if j % log_freq == 0:
            max_idxs = tf.argmax(logits, axis=1)
            # El resultado de lo anterior e un tensor... Por eso necesitamos hacer un numpy()
            acc = np.sum(max_idxs.numpy() == label_batch.numpy()) / len(label_batch.numpy())
            print(f"Iter: {j}, loss={loss:.3f}, accuracy={acc * 100:.3f}%  (Scenario {graph_name})")
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=j)
                tf.summary.scalar('accuracy', acc, step=j)
                # log the gradients
            model_inst.log_gradients(gradients, train_writer, j)
        if acc >= lim_accuracy and loss <= lim_loss:
            print(f'\033[1;31m \nEnd for limit in Accuracy={acc*100:.3f}% and Loss={loss:.3f} \033[0;0m')
            break
    #Accuracy and loss in tests


if __name__ == "__main__":

    #Variables de definición de la red
    #act_funcs = [tf.sigmoid, tf.nn.relu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.softsign]

    num_layers = 6
    act_functions = [tf.nn.tanh,
                     tf.nn.tanh,
                     tf.nn.tanh,
                     tf.nn.tanh,
                     tf.nn.tanh]
    sizes = [300, 100, 10, l_space, 10, 10]
    subfolder_name = "Dense_Model"

    # Construcción de las subfolders

    out_file = f"/TensorFlow_Visualization_{dt.datetime.now().strftime('%d%m%Y%H%M')}"

    # Gardado de datos de entrenamiento

    print(f"\033[1;32m  Running training: \033[0;0m")
    model = mli.ModelGr(activations=act_functions, sizes=sizes,
                            num_layers=num_layers, name=subfolder_name)

    run_training_enc(model, sub_folder=subfolder_name, iterations=5500,
                     lim_accuracy=0.99, lim_loss=0.05,
                     graph_name=subfolder_name, batch_size= 300)

    #Construcción de la media red
    image_batch, label_batch = mli.get_batch(x_validation, y_validation, 17500)
    label_batch = tf.cast(tf.Variable(label_batch), tf.int32)
    Encoder = mli.Encoder(model)
    #Forwarde de la media red
    logits = Encoder.forward(image_batch)
    logits_array = logits.numpy()

    #Accuracy and loss for classifier
    val_logits = model.forward(image_batch)
    max_idxs = tf.argmax(val_logits, axis=1)
    val_loss = model.loss(val_logits, label_batch)
    val_acc = np.sum(max_idxs.numpy() == label_batch.numpy()) / len(label_batch.numpy())
    print(f"\033[1;31m \n End Iter: loss={val_loss:.3f}, accuracy={val_acc * 100:.3f}%. Scenario: {subfolder_name}")

    #Color Codes
    colors = ['#33a02c', '#e31a1c', '#b15928', '#6a3d9a', '#1f78b4',
              '#ff7f00', '#b2df8a', '#fdbf6f', '#fb9a99', '#cab2d6']

    if min(sizes) == 2:

        labeled_images = pd.DataFrame(logits_array, columns=["X", "Y"])
        labeled_images = labeled_images.assign(label=label_batch)

        #Ploteamos el diabolo 2D
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="X", y="Y",
                            hue="label",
                            data=labeled_images,
                            legend="full",
                            palette=colors);
        plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
        plt.title(f'Scatter Latent Space 2D. Python Dense Midpoint\n784 - {sizes}')
        plt.tight_layout()
        plt.show()

    elif min(sizes) == 3:

        labeled_images = pd.DataFrame(logits_array, columns=["X", "Y", "Z"])
        labeled_images = labeled_images.assign(label=label_batch)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(labeled_images["X"], labeled_images["Y"], labeled_images["Z"],
                   c=labeled_images["label"],
                   cmap=mplt.colors.ListedColormap(colors),
                   marker='o',
                   s=5)
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
        plt.title(f'Scatter Latent Space 3D. Python Dense Midpoint\n784 - {sizes}')
        ax.set_xlabel(labeled_images["X"].name)
        ax.set_ylabel(labeled_images["Y"].name)
        ax.set_zlabel(labeled_images["Z"].name)
        plt.show()

    print("\033[1;32m \nWill execute TensorBoard for the last training...")
    print("\nEnd of execution \033[0;0m")
    outfile = STORE_PATH
    print("\ntensorboard --logdir={}/{}\n".format(STORE_PATH, subfolder_name))
    os.system("tensorboard --logdir={}/{}".format(STORE_PATH, subfolder_name))