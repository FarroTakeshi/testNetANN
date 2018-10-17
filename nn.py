# Back-Propagation Neural Networks

import random
import pandas as pd
import numpy

import rnaTraining

random.seed(0)


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    #return math.tanh(x)
    return 1.0 / (1.0 + numpy.e ** -x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    #return 1.0 - y**2
    return y * (1.0 - y)


# our activation function used in output layer is ReLu
def relu(x):
    return numpy.maximum(x, 0)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def drelu(y):
    return (y > 0) * 1


# read csv for training and testing
def readCsv(fileName):
    df = pd.read_csv('C:/Users/takeshi/Documents/Tesis2/' + fileName + '.csv')

    X = df.iloc[:, 1:-1]
    x_values = X.values

    y = df.iloc[:, 9]
    y_values = y.values

    arr = []
    for x in range(x_values.__len__()):
        arr.append([x_values[x], [y_values[x]]])

    return arr


class NN:
    def __init__(self, ni, nh1, nh2, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh1 = nh1
        self.nh2 = nh2
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah1 = [1.0] * self.nh1
        self.ah2 = [1.0] * self.nh2
        self.ao = [1.0] * self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh1)
        self.wh = makeMatrix(self.nh1, self.nh2)
        self.wo = makeMatrix(self.nh2, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh1):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh1):
            for k in range(self.nh2):
                self.wh[j][k] = rand(-0.2, 0.2)
        for k in range(self.nh2):
            for l in range(self.no):
                self.wo[k][l] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh1)
        self.ch = makeMatrix(self.nh1, self.nh2)
        self.co = makeMatrix(self.nh2, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = sigmoid(inputs[i])
            #self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah1[j] = sigmoid(sum)
            #self.ah[j] = leakrelu(sum)

        # hidden activations
        for j in range(self.nh2):
            sum = 0.0
            for i in range(self.nh1):
                sum = sum + self.ah1[i] * self.wh[i][j]
            self.ah2[j] = sigmoid(sum)
            #self.ah[j] = leakrelu(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh2):
                sum = sum + self.ah2[j] * self.wo[j][k]
            #self.ao[k] = sigmoid(sum)
            self.ao[k] = relu(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = drelu(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas2 = [0.0] * self.nh2
        for j in range(self.nh2):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas2[j] = dsigmoid(self.ah2[j]) * error

        # calculate error terms for hidden
        hidden_deltas1 = [0.0] * self.nh1
        for j in range(self.nh1):
            error = 0.0
            for k in range(self.no):
                error = error + hidden_deltas2[k] * self.wh[j][k]
            hidden_deltas1[j] = dsigmoid(self.ah1[j]) * error

        # update output weights
        for j in range(self.nh2):
            for k in range(self.no):
                change = output_deltas[k] * self.ah2[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.nh1):
            for j in range(self.nh2):
                change = hidden_deltas2[j] * self.ah1[i]
                self.wh[i][j] = self.wh[i][j] + N * change + M * self.ch[i][j]
                self.ch[i][j] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh1):
                change = hidden_deltas1[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            result = int(round(numpy.array(self.update(p[0])) * 1000.0))
            real_effort = int(numpy.array(p[1]) * 1000.0)
            print(p[0], '->', result, 'Real Effort: ', real_effort,
                  'Error: ' + str(((result - real_effort) / float(real_effort)) * 100.0) + '%')

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Hidden weights:')
        for j in range(self.nh1):
            print(self.wh[j])
        print()
        print('Output weights:')
        for j in range(self.nh2):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.9):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)

        rnaTraining.insertTraining(self.wi, self.wh, self.wo, self.ni, self.nh1, self.nh2, self.no)

    def estimate(self, inputs):
        hidden1_trained = rnaTraining.getFirstHiddenWeights()
        hidden2_trained = rnaTraining.getSecondHiddenWeights()
        output_trained = rnaTraining.getOutputWeights()

        for i in range(hidden1_trained.__len__()):
            self.wi[i] = hidden1_trained[i]
        for j in range(hidden2_trained.__len__()):
            self.wi[j] = hidden1_trained[j]
        for k in range(output_trained.__len__()):
            self.wo[k] = output_trained[k]
        estimation = int(round(numpy.array(self.update(inputs)) * 1000.0))
        print ('Estimation: ' + str(estimation))


def demo():
    pat_train = readCsv('UCP_Dataset_train_v2')
    pat_test = readCsv('UCP_Dataset_test2_1')

    # create a network with two input, two hidden, and one output nodes
    n = NN(8, 4, 2, 1)

    # train it with some patterns
    n.train(pat_train)
    # test it
    n.test(pat_test)
    # test estimation
    n.estimate([1, 1, 10, 9, 5, 1, 50, 13])


if __name__ == '__main__':
    demo()