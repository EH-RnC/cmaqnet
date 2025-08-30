"""CMAQNet model class."""

from typing import Tuple
from typing_extensions import Self
import numpy.typing as npt

import pandas as pd
import numpy as np
import tensorflow as tf

from cmaqnet.alloc import allocation
alloc_df = pd.DataFrame(allocation, columns=['Region_Name', 'Region_Code', 'Row', 'Column', 'Ratio'])

@tf.keras.utils.register_keras_serializable(package='CMAQNet')
class GriddingLayer(tf.keras.layers.Layer):
    region_codes = {
        'A': 'Seoul City', 'B': 'Incheon City', 'C': 'Busan City', 'D': 'Daegu City',
        'E': 'Gwangju City', 'F': 'Gyeonggi-do', 'G': 'Gangwon-do', 'H': 'Chungbuk-do',
        'I': 'Chungnam-do', 'J': 'Gyeongbuk-do', 'K': 'Gyeongnam-do', 'L': 'Jeonbuk-do',
        'M': 'Jeonnam-do', 'N': 'Jeju-do', 'O': 'Daejeon City', 'P': 'Ulsan City', 'Q': 'Sejong City'}
    def __init__(self, allocation, **kwargs):
        super().__init__(**kwargs)
        # Check if allocation is a dictionary and convert it to DataFrame
        if isinstance(allocation, dict):
            # Convert dictionary back to DataFrame
            self.allocation = pd.DataFrame(allocation)
        else:
            self.allocation = allocation
        
        # Pre-compute indices and ratios for each region
        self.indices = []
        self.ratios = []
        for key in self.region_codes.keys():
            index = self.allocation.loc[self.allocation['Region_Code'] == key, ['Row', 'Column']].values
            ratio = self.allocation.loc[self.allocation['Region_Code'] == key, ['Ratio']].values / 100
            self.indices.append(tf.constant(index, dtype=tf.int32))
            self.ratios.append(tf.constant(ratio, shape=(1, len(ratio), 1), dtype=tf.float32))

    def call(self, inputs_1):
        reshaped_inputs_1 = tf.reshape(inputs_1, (-1, 17, 7))
        batch_size = tf.shape(inputs_1)[0]
        ctrl_map_shape_batch = (batch_size, 82, 67, 7)
        ctrl_map = tf.zeros(ctrl_map_shape_batch, dtype=tf.float32)
        for i, (index, ratio) in enumerate(zip(self.indices, self.ratios)):
            value = reshaped_inputs_1[:, i:i+1, :] * ratio
            batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, tf.shape(index)[0]])
            full_indices = tf.concat([batch_indices[..., tf.newaxis], tf.tile(index[tf.newaxis, ...], [batch_size, 1, 1])], axis=-1)
            ctrl_map = tf.tensor_scatter_nd_add(ctrl_map, full_indices, value)
        return ctrl_map

    def get_config(self):
        # Serialize allocation as a dictionary for model saving
        config = super().get_config()
        config.update({
            "region_codes": self.region_codes,
            "allocation": self.allocation.to_dict(orient='list')})
        return config

def get_unet_model(
    out_channel:int=1,
    hidden_size:tuple=(128, 96),
    in_filters:int=20,
    time_dims:int=128,
    kernel:int=12,
    activation:str='silu',
    dropout:float=0.0,
    use_abs:bool=False,
    no_time_emb:bool=False) -> tf.keras.Model:
    def attention_gate(x, g, filters:int=1):
        x_map = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
        g_map = tf.keras.layers.Conv2D(filters, 1, padding='same')(g)
        x_map = tf.keras.layers.BatchNormalization()(x_map)
        g_map = tf.keras.layers.BatchNormalization()(g_map)
        x_map = tf.keras.layers.Activation(activation)(x_map)
        g_map = tf.keras.layers.Activation(activation)(g_map)
        psi = tf.keras.layers.Add()([x_map, g_map])
        psi = tf.keras.layers.Conv2D(filters, 1, padding='same')(psi)
        psi = tf.keras.layers.Activation('sigmoid')(psi)
        return psi

    def time_embedding(t, dim:int=128):
        half_dim = dim // 2
        emb = tf.math.log(10000.) / (half_dim - 1)
        emb = tf.math.exp(-tf.range(half_dim, delta=emb))
        emb = t * emb[None, :]
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1)
        emb = tf.keras.layers.Dense(dim)(emb)
        emb = tf.keras.layers.Activation(activation)(emb)
        return emb

    def encoder_block(x_map, t_emb=None, filters:int=1, kernel:int=3, dropout:float=0.0):
        x_map = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x_map)
        x_map = tf.keras.layers.BatchNormalization()(x_map)
        x_map = tf.keras.layers.Activation(activation)(x_map)
        x_param = None
        if t_emb is not None:
            t_emb = tf.keras.layers.Conv2D(filters, kernel, padding='same')(t_emb)
            t_emb = tf.keras.layers.Activation(activation)(t_emb)
            # t_emb = tf.keras.layers.Reshape((1, 1, filters))(t_emb)
            t_emb = tf.keras.layers.Resizing(*x_map.shape[1:3])(t_emb)
            x_param = tf.keras.layers.Multiply()([x_map, t_emb])
        x_out = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x_map)
        if x_param is not None:
            x_out = tf.keras.layers.Add()([x_out, x_param])
        x_out = tf.keras.layers.BatchNormalization()(x_out)
        x_out = tf.keras.layers.Activation(activation)(x_out)
        x_out = tf.keras.layers.Dropout(dropout)(x_out)
        return x_out
    
    def decoder_block(x0, x1, filters:int=1, kernel:int=3, dropout:float=0.0, t_emb=None):
        x0 = tf.keras.layers.Conv2DTranspose(
            filters, kernel, strides=2, padding='same')(x0)
        x0 = tf.keras.layers.BatchNormalization()(x0)
        x0 = tf.keras.layers.Activation(activation)(x0)
        x_param = None
        if t_emb is not None:
            t_emb = tf.keras.layers.Conv2D(filters, kernel, padding='same')(t_emb)
            t_emb = tf.keras.layers.Activation(activation)(t_emb)
            # t_emb = tf.keras.layers.Reshape((1, 1, filters))(t_emb)
            t_emb = tf.keras.layers.Resizing(*x0.shape[1:3])(t_emb)
            x_param = tf.keras.layers.Multiply()([x0, t_emb])
        x0 = tf.keras.layers.Concatenate()([x0, x1])
        x0 = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x0)
        if x_param is not None:
            x0 = tf.keras.layers.Add()([x0, x_param])
        x0 = tf.keras.layers.BatchNormalization()(x0)
        x0 = tf.keras.layers.Activation(activation)(x0)
        x0 = tf.keras.layers.Dropout(dropout)(x0)
        return x0
    
    ctrl_inputs = tf.keras.Input(shape=(119,))
    met_inputs = tf.keras.Input(shape=(82, 67, 7))
    
    # met_layer = MatLayer(met_data)

    if no_time_emb:
        t_emb = None
        ctrl_maps = GriddingLayer(alloc_df)(ctrl_inputs)
        inputs = [ctrl_inputs, met_inputs]
    else:
        time_inputs = tf.keras.Input(shape=(1,))
        ctrl_maps = GriddingLayer(alloc_df)(ctrl_inputs)
        t_emb = time_embedding(time_inputs, time_dims)
        inputs = [ctrl_inputs, met_inputs, time_inputs]
    
    # ctrl_maps = tf.keras.layers.concatenate([ctrl_maps, met_inputs], axis=-1)
    ctrl_maps = tf.keras.layers.Resizing(*hidden_size)(ctrl_maps)
    met_maps = tf.keras.layers.Resizing(*hidden_size)(met_inputs)
    
    x = x0 = encoder_block(ctrl_maps, met_maps, in_filters, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    x = x1 = encoder_block(x, met_maps, in_filters*2, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    x = x2 = encoder_block(x, met_maps, in_filters*4, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    x = x3 = encoder_block(x, met_maps, in_filters*8, kernel, dropout)
    x = tf.keras.layers.MaxPool2D()(x)
    
    x = encoder_block(x, met_maps, in_filters*16, kernel, dropout)
    x = decoder_block(x, x3, in_filters*8, kernel, dropout, met_maps)
    x = decoder_block(x, x2, in_filters*4, kernel, dropout, met_maps)
    x = decoder_block(x, x1, in_filters*2, kernel, dropout, met_maps)
    x = decoder_block(x, x0, in_filters, kernel, dropout, met_maps)
    
    x = tf.keras.layers.Resizing(82, 67)(x)
    x = tf.keras.layers.Conv2D(out_channel, 1)(x)
    if use_abs:
        x = tf.keras.layers.Activation('relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def load_model(model_path:str) -> tf.keras.Model:
    unet_loaded = tf.saved_model.load(model_path).signatures["serving_default"]
    ctrl_inputs = tf.keras.Input(shape=(119,))
    time_inputs = tf.keras.Input(shape=(1,))
    inputs = [ctrl_inputs, time_inputs]
    outputs = unet_loaded(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    