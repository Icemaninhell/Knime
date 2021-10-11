import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import ml_functions as mli
import datetime as dt
import os

# Obtención de Dataset

STORE_PATH = f"/home/isra/PycharmProjects/CodingTensorFlowV2/StoredResults/CH4_Gradient_descent/{dt.datetime.now().strftime('%d%m%Y%H%M')}"

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Datasets capturados


# Función de entrenamiento
def run_training(model_inst: mli.Model, sub_folder: str, iterations: int = 2500,
                 batch_size: int = 32, log_freq: int = 200, lim_accuracy: float = 0.99,
                 lim_loss: float = 0.05, graph_name: str = None):

    # Directorio para almacenar datos
    train_writer = tf.summary.create_file_writer(STORE_PATH + "/" + sub_folder)
    # Guardamos el grafo
    model_inst.plot_computational_graph(train_writer, x_train[:batch_size, :, :], graph_name)
    # Selección del optimizador
    optimizer = tf.keras.optimizers.Adam()  # Puede tener un argumento de learning rate
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
        if acc >= lim_accuracy or loss <= lim_loss:
            break


if __name__ == "__main__":

    cicle_one = False

    if cicle_one:
        scenarios_names = ["tan_hip"]
        act_funcs = [tf.nn.tanh]
    else:
        scenarios_names = ["sigmoid", "relu", "leaky_relu",
                           "tan_hip", "soft_plus", "soft_sign"]
        act_funcs = [tf.sigmoid, tf.nn.relu, tf.nn.leaky_relu,
                     tf.nn.tanh, tf.nn.softplus, tf.nn.softsign]

    # Construcción de las subfolders
    scenarios = scenarios_names
    out_file = f"/TensorFlow_Visualization_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
    scenarios = [x + out_file for x in scenarios]

    assert len(scenarios) == len(act_funcs)
    # Gardado de datos de entrenamiento
    for i in range(len(scenarios)):
        print(f"\033[1;32m  Running scenario: {scenarios[i]} \033[0;0m")
        model = mli.Model(act_funcs[i], num_layers=6, hidden_size=10, outer_size=10, name=scenarios_names[i])
        run_training(model, sub_folder=scenarios[i], iterations=4000, lim_accuracy=0.98, lim_loss=0.05,
                     graph_name=scenarios_names[i])

    ans = input("\033[1;32m \nChoose last(l) or all (a) or else for no Tensor Board execution: ")
    #https://stackabuse.com/how-to-print-colored-text-in-python/
    if ans == "a":
        print("\033[1;32m \nWill execute TensorBoard for all scenarios...")
        print("\nEnd of execution \033[0;0m")
        outfile = STORE_PATH
        print("\ntensorboard --logdir={}\n".format(STORE_PATH))
        os.system("tensorboard --logdir={}".format(STORE_PATH))
    elif ans == "l":
        print("\033[1;32m \nWill execute TensorBoard for the last scenario...")
        print("\nEnd of execution \033[0;0m")
        outfile = STORE_PATH
        print("\ntensorboard --logdir={}/{}\n".format(STORE_PATH, scenarios[len(scenarios)-1]))
        os.system("tensorboard --logdir={}/{}".format(STORE_PATH, scenarios[len(scenarios)-1]))
