import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


# n_actions = number of actions, fc1_dims and fc2_dims mean fully connected dims which for 1st and 2nd layers consecutively 

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu') # 1st fully connected layer
        self.fc2 = Dense(fc2_dims, activation='relu') # 2nd fully connected layer
        self.fc3 = Dense(n_actions, activation='softmax') # Final layer

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q = self.q(x)

        return q