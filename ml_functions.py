# Compilación de funciones y clases

import tensorflow as tf
import numpy as np


# Toma vectores aleatorios de un grupo
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs, :, :], y_data[idxs]

# Definición de CLASE Sencilla
# Definición de la clase. En la clase se define un objeto y las funciones que le aplican
# así como las estructuras de datos.
class Model(object):
    # Dentro de la clase, se define la función __init__ que sirve en el momento de la instanciación.
    def __init__(self, activation, num_layers=6, hidden_size=10, outer_size=10, name=None):
        # Atención no se define el tamaño de la capa de recepción.
        # No estamos definiendo las matrices de transición sino las capas.
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.nn_model = tf.keras.Sequential(name=f'sequential_{name}')
        for i in range(num_layers):
            self.nn_model.add(tf.keras.layers.Dense(units=hidden_size, activation=activation, name=f'Layer{i + 1}_{name}'))
            # Range solo va hasta el n-1. Es decir, no se crea la última capa
            # Entregamos los logits y ya hacemos como en el ejemplo anterior el softmax y la entropía
        self.nn_model.add(tf.keras.layers.Dense(units=outer_size, name=f'Model_{name}_Out_Layer'))

    @tf.function()
    def forward(self, input_images):
        input_images = tf.cast(input_images, tf.float32)
        input_images = tf.reshape(input_images, [input_images.shape[0], -1])
        input_images = input_images / 255.0
        # Ahora viene el paso de la red. Llamamos al modelo de la función __init__
        logits = self.nn_model(input_images)
        return logits

    @staticmethod
    def loss(logits, labels):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return loss

    def log_gradients(self, gradients, train_writer, step):
        # La siguiente línea es una función de prevención. Verificamos que se
        # cumple la condición
        assert len(gradients) == len(self.nn_model.trainable_variables)
        # Para cada layer tenemos dos funciones que se tienen que entrenar
        for i in range(len(gradients)):
            # Este if está presente porque sólo nos interesan los PESOS Wi que acompañan en su nombre
            # la palabra 'kernel'
            if 'kernel' in self.nn_model.trainable_variables[i].name:
                with train_writer.as_default():
                    tf.summary.scalar(f"mean_{int((i - 1) / 2)}", tf.reduce_mean(tf.abs(gradients[i])), step=step)
                    tf.summary.histogram(f"histogram_{int((i - 1) / 2)}", gradients[i], step=step)
                    tf.summary.histogram(f"hist_weights_{int((i - 1) / 2)}", self.nn_model.trainable_variables[i],
                                         step=step)

    def plot_computational_graph(self, train_writer, x_batch, graph_name):
        # Almacenamiento del grafo
        tf.summary.trace_on(graph=True)
        # Llamada al modelo
        self.forward(x_batch)
        with train_writer.as_default():
            tf.summary.trace_export(name=graph_name, step=0)

# Fin CLASE

# Clase de Modelo Genérico
#D efinición de clase general con vector de entrada dense para las layers
class ModelGr(object):
    # Dentro de la clase, se define la función __init__ que sirve en el momento de la instanciación.
    def __init__(self, activations, sizes, num_layers=6, name=None):

        assert len(activations) >= num_layers-1
        assert len(activations) <= num_layers
        assert len(sizes) == num_layers

        # Atención no se define el tamaño de la capa de recepción.
        # No estamos definiendo las matrices de transición sino las capas.

        self.num_layers = num_layers
        self.hidden_size = sizes

        self.nn_model = tf.keras.Sequential(name=f'sequential_{name}')

        for i in range(num_layers-1):
            self.nn_model.add(tf.keras.layers.Dense(units=sizes[i], activation=activations[i], name=f'Layer{i + 1}_{name}'))
            # Range solo va hasta el n-1. Es decir, no se crea la última capa
            # Entregamos los logits y ya hacemos como en el ejemplo anterior el softmax y la entropía

        if len(activations) < num_layers:
            self.nn_model.add(tf.keras.layers.Dense(units=sizes[num_layers-1], name=f'Model_{name}_Out_Layer'))
        elif len(activations) == num_layers:
            self.nn_model.add(tf.keras.layers.Dense(units=sizes[num_layers-1], activation=activations[num_layers-1], name=f'Layer{num_layers}_{name}'))

    @tf.function()
    def forward(self, input_images):
        input_images = tf.cast(input_images, tf.float32)
        input_images = tf.reshape(input_images, [input_images.shape[0], -1])
        input_images = input_images / 255.0
        # Ahora viene el paso de la red. Llamamos al modelo de la función __init__
        logits = self.nn_model(input_images)
        return logits

    @staticmethod
    def loss(logits, labels):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return loss

    # Función de pérdidas RMSE
    @staticmethod
    def RMSE_loss(ypred, ylabel):
        ylabel = tf.cast(ylabel, tf.float32)
        ylabel = tf.reshape(ylabel, [ylabel.shape[0], -1])
        ylabel = ylabel / 255.0

        loss = tf.reduce_mean(tf.square(ypred - ylabel))

        return loss

    def log_gradients(self, gradients, train_writer, step):
        # La siguiente línea es una función de prevención. Verificamos que se
        # cumple la condición
        assert len(gradients) == len(self.nn_model.trainable_variables)
        # Para cada layer tenemos dos funciones que se tienen que entrenar
        for i in range(len(gradients)):
            # Este if está presente porque sólo nos interesan los PESOS Wi que acompañan en su nombre
            # la palabra 'kernel'
            if 'kernel' in self.nn_model.trainable_variables[i].name:
                with train_writer.as_default():
                    tf.summary.scalar(f"mean_{int((i - 1) / 2)}", tf.reduce_mean(tf.abs(gradients[i])), step=step)
                    tf.summary.histogram(f"histogram_{int((i - 1) / 2)}", gradients[i], step=step)
                    tf.summary.histogram(f"hist_weights_{int((i - 1) / 2)}", self.nn_model.trainable_variables[i],
                                         step=step)

    def plot_computational_graph(self, train_writer, x_batch, graph_name):
        # Almacenamiento del grafo
        tf.summary.trace_on(graph=True)
        # Llamada al modelo
        self.forward(x_batch)
        with train_writer.as_default():
            tf.summary.trace_export(name=graph_name, step=0)

# Fin CLASE

# Clase para la creación de un Encoder desde un Sequential (Supuesto AENC)
class Encoder(object):

    def __init__(self, SeqModel: ModelGr):
        self.nn_model = tf.keras.Sequential(name='Autoencoder')
        for i in range(SeqModel.num_layers):
            aux_layer = SeqModel.nn_model.get_layer(index=i)
            self.nn_model.add(aux_layer)
            if SeqModel.hidden_size[i] == min(SeqModel.hidden_size):
                break

    def forward(self, input_images):
        input_images = tf.cast(input_images, tf.float32)
        input_images = tf.reshape(input_images, [input_images.shape[0], -1])
        input_images = input_images / 255.0
        # Ahora viene el paso de la red. Llamamos al modelo de la función __init__
        logits = self.nn_model(input_images)
        return logits

#FIN CLASE



