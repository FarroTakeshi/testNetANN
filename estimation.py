import nn
import sys
import json

#inputs = sys.argv[1]
#data = json.loads(inputs)

n = nn.NN(8, 4, 2, 1)
result = n.estimate([2, 0, 0, 5, 0, 0, 0.55, 1])
print result