import modules
import initial
import reshape
import sigmod
import propogate
import predict
import optimize
import model

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 50000, learning_rate = 0.0001, print_cost = True)

import learning_rate
import custom_image
