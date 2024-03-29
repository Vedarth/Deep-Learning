import packages
import initialize_parameters
import initialize_parameters_deep
import linear_forward
import linear_activation_forward
import l_model_forward
import linear_backward
import linear_activation_backward
import l_model_backward
import compute_cost

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] =parameters["W" + str(l+1)]  - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] =parameters["b" + str(l+1)]  - learning_rate*grads["db" + str(l+1)]
    ### END CODE HERE ###
    return parameters

def test():
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)
    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))
