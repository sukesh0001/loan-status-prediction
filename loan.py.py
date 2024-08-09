import numpy as np
import pickle
#loading the saved model
loaded_model=pickle.load(open("C:/Users\A S U S/Documents/trained_model.sav",'rb'))
# prompt: make a predictive system

import numpy as np
# Predictive System
input_data = (1,1,0,	1,0,5849,0.0,128.0,360.0,1.0,1)

# Changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('The person is not eligible for loan')
else:
  print('The person is eligible for loan')



