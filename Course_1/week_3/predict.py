import packages
import dataset
import initialize_parameters
import layer_sizes
import forward_propagation
import compute_cost.py
import backward_propagation.py
import update_parameters.py


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X,parameters)
    predictions = A2>0.5
    ### END CODE HERE ###
    
    return predictions

def main():
    parameters, X_assess = predict_test_case()
    predictions = predict(parameters, X_assess)
    print("predictions mean = " + str(np.mean(predictions)))
