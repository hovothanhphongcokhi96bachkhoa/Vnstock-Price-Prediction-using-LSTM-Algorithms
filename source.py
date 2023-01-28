#DECLARATION OF NECESSARY LIBRARY
import numpy as np
import tensorflow as tf
from vnstock import * # Retrieve functions and stock codes in vnstock

#CONSTRUCTION STOCK SURVEY TIME INITIALIZATION
start_date_observation = "01/01/2021"   #Start
end_date_observation   = "01/07/2022"   #End
print ('Day Start: ', start_date_observation)
print ('Day End: '  , end_date_observation  )

#SEE ALL COMPANY DATA ON VNSTOCK
list_of_total_investment_in_vnstock = listing_companies()
print ('List of Companies in VNSTOCK')
print (list_of_total_investment_in_vnstock)
print ('There are 1630 companies in VNSTOCK')
print ('3 stocks in three different sectors')
print ('Choose 1 company in service field, 1 company in construction field and 1 company in transportation field')

#CHOOSE COMPANY ON CONSTRUCTION: Deo Ca Transport Infrastructure Investment JSC with stock code: HHV
HHV = ticker_overview('HHV')      #Overview of information of stock with code: HHV
print ('Information about Deo Ca Transport Infrastructure Investment Joint Stock Company with stock code: HHV')
print (HHV)

#SELECTION OF TRANSPORTATION COMPANY: SAFI Transport Agency Joint Stock Company with stock code: SFI
SFI = ticker_overview('SFI')      #Overview of information of stock with code: SFI
print ('Information about SAFI Transport Agency Joint Stock Company with stock code: SFI')
print (SFI)

#SELECTION OF SERVICES COMPANY: PetroVietnam General Services Corporation with stock code: PET
PET = ticker_overview('PET')       #Overview of information of stock with code: PET
print ('Information about Petroleum General Services Joint Stock Company with stock code: PET')
print (PET)

#Data query of Deo Ca Transport Infrastructure Investment JSC
see_data_HHV =  stock_historical_data(symbol='HHV', start_date="2021-01-01", end_date='2022-07-01')
print ('Stock transaction data of Deo Ca Transport Infrastructure Investment JSC')
print(see_data_HHV)

#Query the company's data SAFI Transport Agency Joint Stock Company
see_data_SFI =  stock_historical_data(symbol='SFI', start_date="2021-01-01", end_date='2022-07-01')
print ('Securities transaction data of SAFI . Transport Agency JSC')
print(see_data_SFI)

#Query the company's data General Petroleum Services Corporation
see_data_PET =  stock_historical_data(symbol='PET', start_date="2021-01-01", end_date='2022-07-01')
print ('Securities trading data of PetroVietnam General Services Corporation')
print(see_data_PET)

#Transaction information by time of day of Deo Ca Transport Infrastructure Investment JSC
stock_in_a_day_data_HHV = stock_intraday_data (symbol='HHV',page_num=0,page_size=6000)
print ('Transaction information by time of day of Deo Ca Transport Infrastructure Investment JSC')
print (stock_in_a_day_data_HHV)

#Transaction information by time of day of SAFI Transport Agency JSC
stock_in_a_day_data_SFI = stock_intraday_data (symbol='SFI',page_num=0,page_size=6000)
print ('Transaction information by time of day of SAFI Transport Agency JSC')
print (stock_in_a_day_data_SFI)

#Trading information by time of day of PetroVietnam General Services Corporation
stock_in_a_day_data_PET = stock_intraday_data (symbol='PET',page_num=0,page_size=6000)
print ('Trading information by time of day of PetroVietnam General Services Corporation')
print (stock_in_a_day_data_PET)

#SORT DATA REFER TO TIME (DAY)
HHV_data_exploration = see_data_HHV.sort_values ('TradingDate')    #HHV
print ('Company data with HHV code after sorting by time')
print (HHV_data_exploration)
SFI_data_exploration = see_data_SFI.sort_values ('TradingDate')    #SFI
print ('Company data with SFI code sorted by time')
print (SFI_data_exploration)
PET_data_exploration = see_data_PET.sort_values ('TradingDate')    #PET
print ('Company data with PET code sorted by time')
print (PET_data_exploration)

#HHV DATA MODELING
import matplotlib.pyplot as plt

plt.plot (HHV_data_exploration.Open.values ,color = 'blue' , label = 'Open',scaley = True,data = None,marker ='*',linestyle='dashed',
          linewidth=1)
plt.plot (HHV_data_exploration.Close.values,color = 'red'  , label = 'Close',scaley = True,data = None,marker ='+',linestyle='dashed',
          linewidth=1)
plt.plot (HHV_data_exploration.High.values ,color = 'green', label = 'High',scaley = True,data = None,marker ='o',linestyle='dashed',
          linewidth=1)
plt.plot (HHV_data_exploration.Low.values  ,color = 'black', label = 'Low', scaley = True,data = None,marker ='.',linestyle='dashed',
          linewidth=1)
plt.title ('Stock transaction data of Deo Ca Transport Infrastructure Investment JSC from 2021-01-01 to 2022-07-01'
           , color = 'red',fontstyle='italic')
plt.xlabel ('Days')
plt.ylabel ('Price')
plt.legend (loc = 'lower right')
plt.show ()

#SFI DATA MODELING
plt.plot (SFI_data_exploration.Open.values  , color = 'green', label = 'Open',scaley = True,data = None,marker ='.',linestyle='dashed',
          linewidth=1)
plt.plot (SFI_data_exploration.Close.values , color = 'red'  , label = 'Close',scaley = True,data = None,marker ='o',linestyle='dashed',
          linewidth=1)
plt.plot (SFI_data_exploration.High.values  , color = 'blue' , label = 'High',scaley = True,data = None,marker ='^',linestyle='dashed',
          linewidth=1)
plt.plot (SFI_data_exploration.Low.values   , color = 'black', label = 'Low',scaley = True,data = None,marker ='1',linestyle='dashed',
          linewidth=1)
plt.title ('Stock trading data of SAFI Transport Agency JSC from 2021-01-01 to 2022-07-01'
           , color = 'green',fontstyle='italic')
plt.xlabel ('Days')
plt.ylabel ('Price')
plt.legend (loc = 'lower right')
plt.show ()

#PET DATA MODELING
plt.plot (PET_data_exploration.Open.values  , color = 'green', label = 'Open',scaley = True,data = None,marker ='.',linestyle='dashed',
          linewidth=1)
plt.plot (PET_data_exploration.Close.values , color = 'red'  , label = 'Close',scaley = True,data = None,marker ='^',linestyle='dashed',
          linewidth=1)
plt.plot (PET_data_exploration.High.values  , color = 'blue' , label = 'High',scaley = True,data = None,marker ='<',linestyle='dashed',
          linewidth=1)
plt.plot (PET_data_exploration.Low.values   , color = 'black', label = 'Low',scaley = True,data = None,marker ='o',linestyle='dashed',
          linewidth=1)
plt.title ('Securities trading data of PetroVietnam General Services Corporation from 2021-01-01 to 2022-07-01'
           , color = 'blue',fontstyle='italic')
plt.xlabel ('Days')
plt.ylabel ('Price')
plt.legend (loc = 'best')
plt.show ()

#USED DATA REQUIREMENT TO BUILD PROSPECTIVE MODELS
#we do not need collumn named 'TradingDate' and 'Volume', so I deleted it in order not to miss data
#with axis = 1 yeild to column, and '0' yeild to variables
HHV_used_data = HHV_data_exploration.drop (['TradingDate','Volume'], axis = 1, level = None,
                                           inplace = False , errors = 'ignore') #Deleted useless column in dataset
SFI_used_data = SFI_data_exploration.drop (['TradingDate','Volume'], axis = 1, level = None,
                                           inplace = False , errors = 'ignore') #Deleted useless column in dataset
PET_used_data = PET_data_exploration.drop (['TradingDate','Volume'], axis = 1, level = None,
                                           inplace = False , errors = 'ignore') #Deleted useless column in dataset

#NORMALIZATION FOR HHV
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

#We create function to scale data, the return is 4 terms 'Open','High','Low','Close'
def normalize_data_HHV (dementionality_reduction_HVH):
    #rescale back to -1 and 1 with 1 is highest data source and -1 is lowest data
    scaler_HHV = MinMaxScaler ()
    dementionality_reduction_HVH ['Open' ] = scaler_HHV.fit_transform (dementionality_reduction_HVH.Open.values.reshape (-1,1))
    dementionality_reduction_HVH ['High' ] = scaler_HHV.fit_transform (dementionality_reduction_HVH.High.values.reshape (-1,1))
    dementionality_reduction_HVH ['Low'  ] = scaler_HHV.fit_transform (dementionality_reduction_HVH.Low.values.reshape  (-1,1))
    dementionality_reduction_HVH ['Close'] = scaler_HHV.fit_transform (dementionality_reduction_HVH.Close.values.reshape(-1,1))
    return dementionality_reduction_HVH
HHV_data_exploration_normalized = normalize_data_HHV(HHV_used_data)

#NORMALIZATION FOR SFI
def normalize_data_SFI (dementionality_reduction_SFI):
    scaler_SFI = MinMaxScaler ()
    dementionality_reduction_SFI ['Open'] = scaler_SFI.fit_transform  (dementionality_reduction_SFI.Open.values.reshape (-1,1))
    dementionality_reduction_SFI ['High'] = scaler_SFI.fit_transform  (dementionality_reduction_SFI.High.values.reshape (-1,1))
    dementionality_reduction_SFI ['Low']  = scaler_SFI.fit_transform  (dementionality_reduction_SFI.Low.values.reshape  (-1,1))
    dementionality_reduction_SFI ['Close'] = scaler_SFI.fit_transform (dementionality_reduction_SFI.Close.values.reshape(-1,1))
    return dementionality_reduction_SFI
SFI_data_exploration_normalized = normalize_data_SFI (SFI_used_data)

#NORMALIZATION FOR PET
def normalize_data_PET (dementionality_reduction_PET):
    scaler_PET = MinMaxScaler ()
    dementionality_reduction_PET ['Open'] = scaler_PET.fit_transform  (dementionality_reduction_PET.Open.values.reshape (-1,1))
    dementionality_reduction_PET ['High'] = scaler_PET.fit_transform  (dementionality_reduction_PET.High.values.reshape (-1,1))
    dementionality_reduction_PET ['Low']  = scaler_PET.fit_transform  (dementionality_reduction_PET.Low.values.reshape  (-1,1))
    dementionality_reduction_PET ['Close'] = scaler_PET.fit_transform (dementionality_reduction_PET.Close.values.reshape(-1,1))
    return dementionality_reduction_PET
PET_data_exploration_normalized = normalize_data_PET (PET_used_data)

#SPLIT DATA FOR HHV
#Split data is crucial step to figure out the compatibility of the predicted data with the actual data
#In this case, we chose 70% of raw data for training, and from this, we predicted 30% of the left time.
#Finally, we compared our predicted data with real data.

def total_data_HHV (vnstock_data_HHV, lenght_test_HHV):
    # convert the given dataframe stock object to 1D Numpy-array representation
    vnstock_online_material_HHV = vnstock_data_HHV.values
    data_HHV = []
    for alpha in range(len(vnstock_online_material_HHV) - lenght_test_HHV):
        data_HHV.append(vnstock_online_material_HHV [ alpha : alpha + lenght_test_HHV ])
    return data_HHV
def total_train_test_data_HHV (data_train_test_HHV , percentage_of_data_for_training_HHV = 0.7):
   print('The number of used data for the model build process',len(data_train_test_HHV))
   data_HHV = np.asarray(data_train_test_HHV)
   data_size_HHV = len(data_HHV)
   number_of_traning_case_HHV = int(np.floor(percentage_of_data_for_training_HHV*data_size_HHV))
   x_train_HHV = data_HHV [:number_of_traning_case_HHV ,:-1,:]
   y_train_HHV = data_HHV [:number_of_traning_case_HHV ,-1 ,:]
   x_test_HHV  = data_HHV [ number_of_traning_case_HHV:,:-1,:]
   y_test_HHV  = data_HHV [ number_of_traning_case_HHV:,-1 ,:]
   return [x_train_HHV, y_train_HHV, x_test_HHV, y_test_HHV]
length_test_HHV = 5  #5vs1++
data_price_HHV = total_data_HHV (HHV_data_exploration_normalized,length_test_HHV)
x_train_HHV, y_train_HHV, x_test_HHV, y_test_HHV = total_train_test_data_HHV (data_price_HHV, 0.7)
print ('The number of data [days] used for training process')
print('Data of HHV for X-axis training process = ' , x_train_HHV.shape)
print('Data of HHV for Y-axis training process = ' , y_train_HHV.shape)
print ('The number of data [days] used for testing process')
print('Data of HHV for X-axis testing process = '  , x_test_HHV.shape)
print('Data of HHV for Y-axis testing process = '  , y_test_HHV.shape)

#SPLIT DATA FOR SFI
#Split data is crucial step to figure out the compatibility of the predicted data with the actual data
#In this case, we chose 70% of raw data for training, and from this, we predicted 30% of the left time.
#Finally, we compared our predicted data with real data.

def total_data_SFI (vnstock_data_SFI , lenght_test_SFI):
    # convert the given dataframe stock object to 1D Numpy-array representation
    vnstock_online_material_SFI = vnstock_data_SFI.values
    data_SFI = []
    for beta in range(len(vnstock_online_material_SFI) - lenght_test_SFI):
        data_SFI.append(vnstock_online_material_SFI [ beta : beta + lenght_test_SFI ])
    return data_SFI
def total_train_test_data_SFI (data_train_test_SFI , percentage_of_data_for_training_SFI = 0.7):
    print('The number of used data for the model build process', len(data_train_test_SFI))
    data_SFI = np.asarray(data_train_test_SFI)
    data_size_SFI = len(data_SFI)
    number_of_traning_case_SFI = int(np.floor(percentage_of_data_for_training_SFI * data_size_SFI))
    x_train_SFI = data_SFI [:number_of_traning_case_SFI, :-1 , :]
    y_train_SFI = data_SFI [:number_of_traning_case_SFI, -1  , :]
    x_test_SFI  = data_SFI [number_of_traning_case_SFI:, :-1 , :]
    y_test_SFI  = data_SFI [number_of_traning_case_SFI:, -1  , :]
    return [x_train_SFI, y_train_SFI, x_test_SFI, y_test_SFI]
length_test_SFI = 5
data_price_SFI = total_data_SFI (SFI_data_exploration_normalized, length_test_SFI)
x_train_SFI, y_train_SFI, x_test_SFI, y_test_SFI = total_train_test_data_SFI (data_price_SFI, 0.7)
print('The number of data [days] used for training process')
print('Data of SFI for X-axis training process = ', x_train_SFI.shape)
print('Data of SFI for Y-axis training process = ', y_train_SFI.shape)
print('The number of data [days] used for testing process')
print('Data of SFI for X-axis testing process = ', x_test_SFI.shape)
print('Data of SFI for Y-axis testing process = ', y_test_SFI.shape)

#SPLIT DATA FOR PET
#Split data is crucial step to figure out the compatibility of the predicted data with the actual data
#In this case, we chose 70% of raw data for training, and from this, we predicted 30% of the left time.
#Finally, we compared our predicted data with real data.

def total_data_PET (vnstock_data_PET, lenght_test_PET):
    # convert the given dataframe stock object to 1D Numpy-array representation
    vnstock_online_material_PET = vnstock_data_PET.values
    data_PET = []
    for gamma in range(len(vnstock_online_material_PET) - lenght_test_PET):
        data_PET.append(vnstock_online_material_PET [ gamma : gamma + lenght_test_PET ])
    return data_PET
def total_train_test_data_PET (data_train_test_PET , percentage_of_data_for_training_PET = 0.7):
   print('The number of used data for the model build process',len(data_train_test_PET))
   data_PET = np.asarray(data_train_test_PET)
   data_size_PET = len(data_PET)
   number_of_traning_case_PET = int(np.floor(percentage_of_data_for_training_PET * data_size_PET))
   x_train_PET = data_PET [:number_of_traning_case_PET ,:-1,:]
   y_train_PET = data_PET [:number_of_traning_case_PET ,-1 ,:]
   x_test_PET  = data_PET [ number_of_traning_case_PET:,:-1,:]
   y_test_PET  = data_PET [ number_of_traning_case_PET:,-1 ,:]
   return [x_train_PET, y_train_PET, x_test_PET, y_test_PET]
length_test_PET = 5
data_price_PET = total_data_PET (PET_data_exploration_normalized,length_test_PET)
x_train_PET, y_train_PET, x_test_PET, y_test_PET = total_train_test_data_PET (data_price_PET, 0.7)
print ('The number of data [days] used for training process')
print('Data of PET for X-axis training process = ' , x_train_PET.shape)
print('Data of PET for Y-axis training process = ' , y_train_PET.shape)
print ('The number of data [days] used for testing process')
print('Data of PET for X-axis testing process = '  , x_test_PET.shape)
print('Data of PET for Y-axis testing process = '  , y_test_PET.shape)

#Exam the active function ReLu, Sigmoid, Tanh, Linear, Leaky ReLu
#First of all, consider ReLu
def ReLu(beta):
    if (beta>0):
        return beta
    else:
        return 0
ReLu_function_yeild_x = [beta for beta in range (-10,10)]  #declare the desired x-axis values
ReLu_function_yeild_y = [ReLu(beta) for beta in ReLu_function_yeild_x]  #calculate the y value of the ReLu function
plt.plot (ReLu_function_yeild_x,ReLu_function_yeild_y)
plt.show()

#BUILDING DATA TRAINING MODEL FOR HHV
from keras.layers import LSTM

model_HHV = Sequential ()
#input_shape = [timesteps , n_features]
model_HHV.add(LSTM(units = 60 , input_shape= (x_train_HHV.shape[1],x_train_HHV.shape[1]) ,return_sequences = True))
model_HHV.add(Dropout(0.2))
model_HHV.add(LSTM( 60 , return_sequences = False))
model_HHV.add(Dropout(0.2))
model_HHV.add(Dense(units = 4))

#we use activate function ReLu (Rectified Linear Unit)
#The rectified linear activation function or ReLU is a non-linear function or piecewise linear function that will output the input directly if it is positive
# otherwise, it will output zero.
#It is the most commonly used activation function in neural networks, especially in Convolutional Neural Networks (CNNs) & Multilayer perceptrons.
#It is simple yet it is more effective than it's predecessors like sigmoid or tanh.
#Mathematically, it is expressed as: f(x)=max(0,x)

model_HHV.add(Activation('relu'))
model_HHV.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_checkpoint_callback_for_HHV = tf.keras.callbacks.ModelCheckpoint(filepath='HHV_Stock.h5', verbose=1, save_best_only=True)
model_HHV.fit(x_train_HHV, y_train_HHV, epochs=200, batch_size=10, verbose=1,
                         callbacks=[model_checkpoint_callback_for_HHV], validation_split=0.3)

#BUILDING DATA TRAINING MODEL FOR SFI
model_SFI = Sequential ()
#input_shape = [timesteps , n_features]
model_SFI.add(LSTM(units = 60 , input_shape= (x_train_SFI.shape[1],x_train_SFI.shape[1]) ,return_sequences = True))
model_SFI.add(Dropout(0.2))
model_SFI.add(LSTM( 60 , return_sequences = False))
model_SFI.add(Dropout(0.2))
model_SFI.add(Dense(units = 4))

#we use activate function ReLu (Rectified Linear Unit)
#The rectified linear activation function or ReLU is a non-linear function or piecewise linear function that will output the input directly if it is positive
# otherwise, it will output zero.
#It is the most commonly used activation function in neural networks, especially in Convolutional Neural Networks (CNNs) & Multilayer perceptrons.
#It is simple yet it is more effective than it's predecessors like sigmoid or tanh.
#Mathematically, it is expressed as: f(x)=max(0,x)

model_SFI.add(Activation('relu'))
model_SFI.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_checkpoint_callback_for_SFI = tf.keras.callbacks.ModelCheckpoint(filepath='SFI_Stock.h5', verbose=1, save_best_only=True)
model_SFI.fit(x_train_SFI, y_train_SFI, epochs=200, batch_size=10, verbose=1,
                         callbacks=[model_checkpoint_callback_for_SFI], validation_split=0.3)

#BUILDING DATA TRAINING MODEL FOR PET
model_PET = Sequential ()
#input_shape = [timesteps , n_features]
model_PET.add(LSTM(units = 60 , input_shape= (x_train_PET.shape[1],x_train_PET.shape[1]) ,return_sequences = True))
model_PET.add(Dropout(0.2))
model_PET.add(LSTM( 60 , return_sequences = False))
model_PET.add(Dropout(0.2))
model_PET.add(Dense(units = 4))

#we use activate function ReLu (Rectified Linear Unit)
#The rectified linear activation function or ReLU is a non-linear function or piecewise linear function that will output the input directly if it is positive
# otherwise, it will output zero.
#It is the most commonly used activation function in neural networks, especially in Convolutional Neural Networks (CNNs) & Multilayer perceptrons.
#It is simple yet it is more effective than it's predecessors like sigmoid or tanh.
#Mathematically, it is expressed as: f(x)=max(0,x)

model_PET.add(Activation('relu'))
model_PET.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_checkpoint_callback_for_PET = tf.keras.callbacks.ModelCheckpoint(filepath='PET_Stock.h5', verbose=1, save_best_only=True)
model_PET.fit(x_train_PET, y_train_PET, epochs=200, batch_size=10, verbose=1,
                         callbacks=[model_checkpoint_callback_for_PET], validation_split=0.3)

#FINAL RESULTS FOR HHV MODEL
from keras.models import load_model

value_of_predicting_HHV = 1
model_HHV_result = load_model('HHV_Stock.h5')
y_mu_HHV = model_HHV_result.predict(x_test_HHV)
plt.plot( y_test_HHV[:,value_of_predicting_HHV], color='blue', label='Real Price'    , marker = 'o' , linestyle='dashed')
plt.plot( y_mu_HHV  [:,value_of_predicting_HHV], color='red' , label='Predicted Price' , marker = 'o' , linestyle='dashed')
plt.title('Prediction for HaTangGiaoThongDeoCa. INC in VNSTOCK')
plt.xlabel('Days')
plt.ylabel('Stock Prices yeild to nomalization')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error
#with an aim to evaluate the accuracy of the predicted model, we can use a lot of method such as RSME, MSE,...
#in this case, we chose MSE (Mean squared error)
#In statistics, the mean squared error (MSE)[1] or mean squared deviation (MSD) of an estimator
#An MSE of zero, meaning that the estimator predicts observations of the parameter  with perfect accuracy, is ideal (but typically not possible).

print ('MSE Value for DAUTUHATANGGIAOTHONGDEOA. INC')
print("Steady State Error of HHV Prediction Process yeild to Open index: ")
print(mean_squared_error(y_test_HHV[:,0], y_mu_HHV[ :,0]))
print("Steady State Error of HHV Prediction Process yeild to High index: ")
print(mean_squared_error(y_test_HHV[:,1], y_mu_HHV[ :,1]))
print("Steady State Error of HHV Prediction Process yeild to Low index: ")
print(mean_squared_error(y_test_HHV[:,2], y_mu_HHV[ :,2]))
print("Steady State Error of HHV Prediction Process yeild to Close index: ")
print(mean_squared_error(y_test_HHV[:,3], y_mu_HHV[ :,3]))

#FINAL RESULTS FOR SFI MODEL
value_of_predicting_SFI = 1
model_SFI_result = load_model('SFI_Stock.h5')
y_mu_SFI = model_SFI_result.predict(x_test_SFI)
plt.plot( y_test_SFI[:,value_of_predicting_SFI], color='blue', label='Real Price'    , marker = 'o' , linestyle='dashed')
plt.plot( y_mu_SFI  [:,value_of_predicting_SFI], color='red' , label='Predicted Price' , marker = 'o' , linestyle='dashed')
plt.title('Prediction for CTCP DAILYVANTAI INC in VNSTOCK')
plt.xlabel('Days')
plt.ylabel('Stock Prices yeild to nomalization')
plt.legend(loc='best')
plt.show()
print ('MSE Value for CTCP DAILYVANTAI. INC')
print("Steady State Error of HHV Prediction Process yeild to Open index: ")
print(mean_squared_error(y_test_SFI[:,0], y_mu_SFI[ :,0]))
print("Steady State Error of HHV Prediction Process yeild to High index: ")
print(mean_squared_error(y_test_SFI[:,1], y_mu_SFI[ :,1]))
print("Steady State Error of HHV Prediction Process yeild to Low index: ")
print(mean_squared_error(y_test_SFI[:,2], y_mu_SFI[ :,2]))
print("Steady State Error of HHV Prediction Process yeild to Close index: ")
print(mean_squared_error(y_test_SFI[:,3], y_mu_SFI[ :,3]))

#FINAL RESULTS FOR PET MODEL
value_of_predicting_PET = 1
model_PET_result = load_model('PET_Stock.h5')
y_mu_PET = model_PET_result.predict(x_test_PET)
plt.plot( y_test_PET[:,value_of_predicting_PET], color='blue', label='Real Price'    , marker = 'o' , linestyle='dashed')
plt.plot( y_mu_PET  [:,value_of_predicting_PET], color='red' , label='Predicted Price' , marker = 'o' , linestyle='dashed')
plt.title('Prediction for Tổng Công ty cổ phần Dịch vụ Tổng hợp Dầu khí in VNSTOCK')
plt.xlabel('Days')
plt.ylabel('Stock Prices yeild to nomalization')
plt.legend(loc='best')
plt.show()
print ('MSE Value for Tổng Công ty cổ phần Dịch vụ Tổng hợp Dầu khí. INC')
print("Steady State Error of HHV Prediction Process yeild to Open index: ")
print(mean_squared_error(y_test_PET[:,0], y_mu_PET[ :,0]))
print("Steady State Error of HHV Prediction Process yeild to High index: ")
print(mean_squared_error(y_test_PET[:,1], y_mu_PET[ :,1]))
print("Steady State Error of HHV Prediction Process yeild to Low index: ")
print(mean_squared_error(y_test_PET[:,2], y_mu_PET[ :,2]))
print("Steady State Error of HHV Prediction Process yeild to Close index: ")
print(mean_squared_error(y_test_PET[:,3], y_mu_PET[ :,3]))


