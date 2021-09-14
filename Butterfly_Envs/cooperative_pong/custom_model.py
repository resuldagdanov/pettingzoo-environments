from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


tf1, tf, tfv = try_import_tf()


class MLPModelV2(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name="my_model"):
        super(MLPModelV2, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        input_layer = tf.keras.layers.Input(obs_space.shape, dtype=obs_space.dtype)
        
        conv_1 = tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu")(input_layer)                       
        conv_2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation="relu")(conv_1) 
        conv_3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation="relu")(conv_2)
        
        flatten = tf.keras.layers.Flatten()(conv_3)
        
        dense_1 = tf.keras.layers.Dense(512, activation="relu")(flatten)

        output = tf.keras.layers.Dense(num_outputs, activation=None)(dense_1)

        value_out = tf.keras.layers.Dense(1, activation=None, name="value_out")(dense_1)
        
        self.base_model = tf.keras.Model(input_layer, [output, value_out])
        self.register_variables(self.base_model.variables)
    
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
