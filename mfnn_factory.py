from bike_share_data import BikeShareData
from neural_network  import NeuralNetwork
from layer           import Dense, Sigmoid, Activation
from mfnn_utils      import pddf_to_nparr
from numpy           import squeeze

def create_np_data_sets():
    
    data = BikeShareData()
    data.retrieve()
    data.preprocess()
    df_training, df_validation, df_test = data.create_data_sets()

    inputs             = pddf_to_nparr(df_training[0])
    targets            = pddf_to_nparr(df_training[1])

    validation_inputs  = pddf_to_nparr(df_validation[0])
    validation_targets = pddf_to_nparr(df_validation[1])
    
    test_inputs        = pddf_to_nparr(df_test[0])
    test_targets       = pddf_to_nparr(df_test[1])
    dates              = pddf_to_nparr(df_test[2])
    mean, std          = data.z_factors['cnt']
    
    np_training_set    = (inputs, targets)
    np_validation_set  = (validation_inputs, validation_targets)
    np_test_set        = (test_inputs, squeeze(test_targets), squeeze(dates), mean, std)

    return np_training_set, np_validation_set, np_test_set


def create_mfnn(input_size):
    mfnn = NeuralNetwork()

    mfnn.add(Dense(32, input_size=input_size))
    mfnn.add(Sigmoid())

    mfnn.add(Dense(1))
    mfnn.add(Activation())

    mfnn.compile()
    
    return mfnn


