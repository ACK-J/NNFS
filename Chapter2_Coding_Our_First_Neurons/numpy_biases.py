import numpy as np
inputs = [[1,2,3,2.5],
           [2,5,-1,2],
           [-1.5,2.7,3.3,-.8]]
weights = [ [0.2,0.8,-0.5,1.0],
            [0.5,-.91,.26,-.5],
            [-0.26,-0.27,.17,.87] ]
biases = [2,3,.5]

outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)
