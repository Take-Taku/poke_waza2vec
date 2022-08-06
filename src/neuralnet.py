from tensorflow.keras.layers import Dense, Attention, AdditiveAttention
import tensorflow as tf
import numpy as np

class simpleAttention(tf.keras.models.Model):
    def __init__(self
                , depth: int
                , *args
                , **kwargs):
        '''
        コンストラクタです。
        :param depth: 隠れ層及び出力の次元
        '''
        super().__init__(*args, **kwargs)
        self.attention = AdditiveAttention()
        self.output_dense_layer = Dense(6, use_bias=False, name='output_dense_layer')

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def call(self
            , embedded: tf.Tensor
            ) -> tf.Tensor:
        '''
        モデルの実行を行います。
        :param input: query のテンソル
        :param memory: query に情報を与える memory のテンソル
        '''
        attention_output = self.attention([embedded, embedded])
        print(attention_output)
        return self.output_dense_layer(attention_output)

    @tf.function
    def train_step(self, x, t):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = self.loss_object(t, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(t, predictions)

    @tf.function
    def test_step(self, x, t):
        test_predictions = model(x)
        t_loss = self.loss_object(t, test_predictions)

        self.test_loss(t_loss)
        self.test_accuracy(t, test_predictions)

    def train(self, embed, efforts, epochs=5):
        for _ in range(epochs):
            for em, eff in zip(embed, efforts):
                self.train_step(em, eff)


    

    
if __name__ == '__main__':
    model = simpleAttention(depth=4)
    embed = [[[0, 1, 3], [2, 3, 1], [3, 4, 1]]]
    embed = tf.cast(embed, tf.int64)
    effort =  tf.cast([0, 1, 0.5, 0.3, 0.2, 0.1], tf.int32)
    model.train([embed], effort)
    
