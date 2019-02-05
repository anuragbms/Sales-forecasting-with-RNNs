def gen_mutant():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    import tensorflow as tf
    from tensorflow.python.saved_model import tag_constants
    import pandas as pd
    import numpy as np
    
    
    
    
    
    
    DATAFILE_VALIDATE = '/home/ubuntu/anurag/rnn/data_for_MRs/mock_kaggle_edit_validate_normalise.csv'
    
    TRAINED_MODEL_PATH = '/home/ubuntu/anurag/rnn/savedModel'
    
    TIME_STEPS = 10
    NUMBER_OF_DAYS_TO_FORECAST = 1
    
    BATCH_SIZE = 100
    
    
    
    
    
    
    
    MIN = 0
    RANGE = 542
    
    
    
    
    
    
    data_validate = pd.read_csv(DATAFILE_VALIDATE)
    
    numValidationData = len(data_validate)
    
    validationData_sales = data_validate['sales_add_309'][0:numValidationData]
    
    
    
    
    
    print(len(validationData_sales))
    
    
    
    
    
    validationData_sales_normalised = [(i - MIN) / RANGE for i in validationData_sales]
    
    
    
    
    
    
    validationDataSequence_sales = np.zeros(shape=(((len(validationData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, TIME_STEPS, 2))
    validationDataSequence_sales_target = np.zeros(shape=(((len(validationData_sales) - TIME_STEPS) - NUMBER_OF_DAYS_TO_FORECAST) + 1, NUMBER_OF_DAYS_TO_FORECAST))
    
    start = 0
    for i in range(TIME_STEPS, (len(validationData_sales) - NUMBER_OF_DAYS_TO_FORECAST) + 1):
        validationDataSequence_sales[start,:,0] = validationData_sales_normalised[start:i]
        validationDataSequence_sales_target[start] = validationData_sales_normalised[i:i + NUMBER_OF_DAYS_TO_FORECAST]
        start += 1
    
    
    
    
    
    validationDataSequence_sales_target.shape
    
    
    
    
    
    with tf.Session() as sess:
        print('Loading the model from:', TRAINED_MODEL_PATH)
        tf.saved_model.loader.load(sess=sess, export_dir=TRAINED_MODEL_PATH, tags=[tag_constants.SERVING])
        
        
        
        inputSequence = tf.get_default_graph().get_tensor_by_name('inputSequencePlaceholder:0')
        targetForecast = tf.get_default_graph().get_tensor_by_name('targetPlaceholder:0')
        
        loss = tf.get_default_graph().get_tensor_by_name('loss_comp:0')
        forecast_originalScale = tf.get_default_graph().get_tensor_by_name('forecast_original_scale:0')
        
        startLoc = 0
        totalLoss = 0
        for i in range(0, len(validationDataSequence_sales) // BATCH_SIZE):
            sequence = validationDataSequence_sales[startLoc:startLoc + BATCH_SIZE,:,:]
            target = validationDataSequence_sales_target[startLoc:startLoc + BATCH_SIZE]
            (fcast, ls) = sess.run([forecast_originalScale, loss], feed_dict={inputSequence: sequence, targetForecast: target})
            
            print('first five predictions (original scale):', fcast[0:5])
            print('first five actuals (original scale)    :', (target[0:5] * RANGE) + MIN)
            totalLoss += ls
            startLoc += BATCH_SIZE
        
        if startLoc < len(validationDataSequence_sales):
            sequence = validationDataSequence_sales[startLoc:]
            target = validationDataSequence_sales_target[startLoc:]
            (fcast, ls) = sess.run([forecast_originalScale, loss], feed_dict={inputSequence: sequence, targetForecast: target})
            totalLoss += ls
        
        print('Validation complete. Total loss:', totalLoss)