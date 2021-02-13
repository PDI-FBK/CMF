import logging

from hyperopt import STATUS_OK, STATUS_FAIL
from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import numpy as np

from src.evaluation.common import evaluate
from src.predictive_model.common import PredictionMethods

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'], 1)
    return df


class PredictiveModel:

    def __init__(self, CONF, model_type, train_df, validate_df):
        self.CONF = None
        self.model_type = model_type
        self.config = None
        self.model = None
        self.full_train_df = train_df
        self.train_df = drop_columns(train_df)
        self.train_df_shaped = None
        self.full_validate_df = validate_df
        self.validate_df = drop_columns(validate_df)
        self.validate_df_shaped = None

    def train_and_evaluate_configuration(self, config, target):
        try:
            model = self._instantiate_model(config)

            self._fit_model(model)

            predicted, scores = self._output_model(model=model)

            actual = self.full_validate_df['label']
            result = evaluate(actual, predicted, scores, loss=target)

            return {
                'status': STATUS_OK,
                'loss': - result['loss'],  # we are using fmin for hyperopt
                'exception': None,
                'config': config,
                'model': model,
                'result': result,
            }
        except Exception as e:
            return {
                'status': STATUS_FAIL,
                'loss': 0,
                'exception': str(e)
            }

    def _instantiate_model(self, config):
        if self.model_type == PredictionMethods.RANDOM_FOREST.value:
            model = RandomForestClassifier(**config)
        elif self.model_type == PredictionMethods.LSTM.value:

            # get number of activities??


            self.train_df_shaped = np.reshape((-1, self.CONF['prefix_length'], ))  # reshape from 2D to 3D

            main_input = tf.keras.layers.Input(shape=(self.train_df_shaped[1], self.train_df_shaped[2]), name='main_input')
            b1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=False, dropout=0.2))(main_input)

            num_labels = len(set(self.full_train_df['label'].tolist()))
            out_output = tf.keras.layers.Dense(num_labels, activation='softmax', name='output', kernel_initializer='glorot_uniform')(b1)

            model = tf.keras.models.Model(inputs=[main_input], outputs=[out_output])
            model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam')
            model.summary()


        else:
            raise Exception('unsupported model_type')
        return model

    def _fit_model(self, model):
        if self.model_type == PredictionMethods.RANDOM_FOREST.value:
            model.fit(self.train_df, self.full_train_df['label'])

        elif self.model_type == PredictionMethods.LSTM.value:

            # label matrix
            # exclude hyper-parameters

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                              mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

            label = self.full_train_df['label']


            model.fit(self.train_df_shaped, {'output': label},
                      validation_split=0.1,
                      verbose=1,
                      callbacks=[early_stopping, lr_reducer],
                      batch_size=128,
                      epochs=100)


    def _output_model(self, model):
        if self.model_type == PredictionMethods.RANDOM_FOREST.value:
            predicted = model.predict(self.validate_df)
            scores = model.predict_proba(self.validate_df)[:, 1]

        elif self.model_type == PredictionMethods.LSTM.value:
            predicted = ""
            scores = ""
        else:
            raise Exception('unsupported model_type')

        return predicted, scores
