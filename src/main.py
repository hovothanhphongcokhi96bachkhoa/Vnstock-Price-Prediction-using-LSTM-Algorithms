#DÉCLARATION DE BIBLIOTHÈQUE NÉCESSAIRE
import numpy as np
import tensorflow as tf
from vnstock import * #Récupérer les fonctions et les codes de stock dans vnstock

"""Tout d'abord, nous choisissons le jeu de données VNSTOCK pour enquêter et sélectionner 3 entreprises dans 3 domaines
du transport, de la construction et des services dans toutes les entreprises cotées en bourse. Ici, contrairement au 
fait que nous prédisons la valeur future des actions (valeur de actions qui n'ont jamais été émises sur le marché), il 
sera difficile d'évaluer la compatibilité de l'ensemble de données formé par ce groupe. Dans ce rapport, pour résoudre 
ce problème, les membres de l'équipe ont décidé de choisir une période de temps, en prenant 70% de l'ensemble 
expérimental de données comme base du processus de formation et 30% des données. le reste de l'ensemble de données 
nous servira à créer le modèle prédictif. À partir de ce modèle de prédiction approximatif, nous le comparerons aux 
données de stock réelles disponibles et tirerons des conclusions sur la compatibilité de l'ensemble de données prédites 
par l'erreur quadratique moyenne. De plus, l'équipe approfondira l'état de l'ensemble de données prédit dans chaque 
cas où l'entreprise appartient à 'Under Fitting' ou 'Optimum' ou 'Over Fitting'"""

#INITIALISATION DU TEMPS DE L'ENQUÊTE SUR LE PARC DE CONSTRUCTION
start_date_observation = "01/01/2021"   #Start
end_date_observation   = "01/07/2022"   #End
print ('Day Start: ', start_date_observation)
print ('Day End: '  , end_date_observation  )

#SEE ALL COMPANY DATA ON VNSTOCK

"""Afin de sélectionner les entreprises en fonction des exigences du sujet, il faut d'abord voir toutes les entreprises 
cotées à la bourse Vnstock, nous utilisons la fonction listing.companies de la bibliothèque vnstock pour interroger 
toutes les données des codes boursiers."""

list_of_total_investment_in_vnstock = listing_companies()
print ('List of Companies in VNSTOCK')
print (list_of_total_investment_in_vnstock)
print ('There are 1630 companies in VNSTOCK')
print ('3 stocks in three different sectors')
print ('Choose 1 company in service field, 1 company in construction field and 1 company in transportation field')

#CHOOSE COMPANY ON CONSTRUCTION: Deo Ca Transport Infrastructure Investment JSC with stock code: HHV

"""Nous choisissons l'entreprise de construction comme Deo Ca Transport Infrastructure Investment Joint Stock Company 
avec le code boursier HHV et imprimons les informations du code boursier HHV à l'écran."""

HHV = ticker_overview('HHV')      #Overview of information of stock with code: HHV
print ('Information about Deo Ca Transport Infrastructure Investment Joint Stock Company with stock code: HHV')
print (HHV)

#SELECTION OF TRANSPORTATION COMPANY: SAFI Transport Agency Joint Stock Company with stock code: SFI

"""Nous choisissons la société dans le domaine du transport, qui est SAFI Transport Agency Joint Stock Company avec le 
code de stock SFI et imprimons les informations du code de stock SFI à l'écran."""

SFI = ticker_overview('SFI')      #Overview of information of stock with code: SFI
print ('Information about SAFI Transport Agency Joint Stock Company with stock code: SFI')
print (SFI)

#SELECTION OF SERVICES COMPANY: PetroVietnam General Services Corporation with stock code: PET

"""Nous choisissons la société dans le domaine du service en tant que PetroVietnam General Services Corporation avec le 
code de stock PET et imprimons les informations du code de stock PET à l'écran."""

PET = ticker_overview('PET')       #Overview of information of stock with code: PET
print ('Information about Petroleum General Services Joint Stock Company with stock code: PET')
print (PET)

"""Afin de répondre aux exigences du grand exercice, qui consiste à prédire la valeur du stock pour les 3 prochains 
mois (c'est-à-dire que le nombre de jours de prédiction doit être supérieur à 90 jours). Par conséquent, la période 
de temps requise par le grand exercice n'est pas suffisante (environ 132 jours, soit 132 jeux de données expérimentaux). 
Ainsi, nous formons le problème comme suit: nombre de jours à prédire = 30% multiplié par le nombre total 
d'observations de l'ensemble de données. Le nombre minimum de jours de prédiction étant de 90 jours, nous pouvons 
calculer le nombre d'échantillons d'observation nécessaires et optimaux pour l'ensemble de données dans ce cas : 300 
jours. Étant donné que la période recommandée dans l'exigence du grand exercice est du 1er juillet 2021 au 31 décembre 
2021, il n'y a que 129 ensembles d'observations en moyenne, nous devons donc choisir 2 intervalles de données 
supplémentaires sur une période de six mois. (129x3 = 387 ensembles de données) remplira les conditions minimales pour 
mener une formation. Sur cette base, après discussion, le groupe a décidé de prolonger de six mois avant et après la 
période proposée. La période sélectionnée est donc du 01/01/2021 au 01/07/2022"""

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

"""Commentaire : Après impression à l'écran, on constate que sur la période du 01/01/2021 au 07/01/2022, 3 jeux de 
données des stocks HHV, SFI, PET ont respectivement 362 échantillons observés, 371 et 371. Dans chaque ensemble de 
données, Vnstock renvoie 6 tableaux de données, y compris Ouvert [Prix d'ouverture]: est le prix de clôture de la 
session précédente], Élevé [Prix le plus élevé: est le prix le plus élevé de la session précédente] dans une séance 
de négociation ou dans un cycle de surveillance des mouvements de prix ], Bas [Prix bas: est le prix le plus bas 
d'une séance de négociation ou d'un cycle de suivi des prix], Clôture [Prix de clôture: est le prix d'exercice au 
dernier ordre de correspondance du jour de négociation], Volume [Volume de négociation], Date de négociation 
[Date de négociation]. Ensuite, nous utilisons la fonction intraday incluse dans la bibliothèque vnstock pour 
approfondir les données boursières sur 1 jour de 3 actions HHV, SFI et PET."""

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

"""Un problème que les programmeurs en science des données doivent garder à l'esprit est le moment du modèle 
d'observation. Dans l'ensemble de données de vnstock déjà trié par ordre croissant de date, nous n'avons 
fondamentalement pas besoin d'intervenir dans ce problème, mais de prendre l'habitude de traiter les données pour les 
programmeurs. Avec les autres ensembles de données brutes, l'équipe a décidé de nouveau trier la date de l'échantillon 
d'observations. Si l'ordre des dates est inversé, le processus d'entraînement, de test et d'évaluation du modèle de 
prédiction de l'algorithme Long-Short-Term-Memory n'aura plus de sens. Nous utilisons la fonction sort.value en python 
pour trier"""

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

plt.plot (HHV_data_exploration.Open.values ,color = 'blue' , label = 'Open',
          scaley = True,data = None,marker ='*',linestyle='dashed',linewidth=1)
plt.plot (HHV_data_exploration.Close.values,color = 'red'  , label = 'Close',
          scaley = True,data = None,marker ='+',linestyle='dashed',linewidth=1)
plt.plot (HHV_data_exploration.High.values ,color = 'green', label = 'High',
          scaley = True,data = None,marker ='o',linestyle='dashed',linewidth=1)
plt.plot (HHV_data_exploration.Low.values  ,color = 'black', label = 'Low',
          scaley = True,data = None,marker ='.',linestyle='dashed',linewidth=1)
plt.title ('Stock transaction data of Deo Ca Transport Infrastructure Investment JSC from 2021-01-01 to 2022-07-01'
           , color = 'red',fontstyle='italic')
plt.xlabel ('Days')
plt.ylabel ('Price')
plt.legend (loc = 'lower right')
plt.show ()

#SFI DATA MODELING
plt.plot (SFI_data_exploration.Open.values  , color = 'green', label = 'Open',
          scaley = True,data = None,marker ='.',linestyle='dashed',linewidth=1)
plt.plot (SFI_data_exploration.Close.values , color = 'red'  , label = 'Close',
          scaley = True,data = None,marker ='o',linestyle='dashed',linewidth=1)
plt.plot (SFI_data_exploration.High.values  , color = 'blue' , label = 'High',
          scaley = True,data = None,marker ='^',linestyle='dashed',linewidth=1)
plt.plot (SFI_data_exploration.Low.values   , color = 'black', label = 'Low',
          scaley = True,data = None,marker ='1',linestyle='dashed',linewidth=1)
plt.title ('Stock trading data of SAFI Transport Agency JSC from 2021-01-01 to 2022-07-01'
           , color = 'green',fontstyle='italic')
plt.xlabel ('Days')
plt.ylabel ('Price')
plt.legend (loc = 'lower right')
plt.show ()

#PET DATA MODELING
plt.plot (PET_data_exploration.Open.values  , color = 'green', label = 'Open',
          scaley = True,data = None,marker ='.',linestyle='dashed',linewidth=1)
plt.plot (PET_data_exploration.Close.values , color = 'red'  , label = 'Close',
          scaley = True,data = None,marker ='^',linestyle='dashed',linewidth=1)
plt.plot (PET_data_exploration.High.values  , color = 'blue' , label = 'High',
          scaley = True,data = None,marker ='<',linestyle='dashed',linewidth=1)
plt.plot (PET_data_exploration.Low.values   , color = 'black', label = 'Low',
          scaley = True,data = None,marker ='o',linestyle='dashed',linewidth=1)
plt.title ('Securities trading data of PetroVietnam General Services Corporation from 2021-01-01 to 2022-07-01'
           , color = 'blue',fontstyle='italic')
plt.xlabel ('Days')
plt.ylabel ('Price')
plt.legend (loc = 'best')
plt.show ()

"""L'ensemble de données que VNSTOCK nous fournit comprend 6 dimensions, mais nous avons une colonne de données que 
nous ne considérerons pas dans ce grand problème d'exercice, qui est la colonne 'TradingDate' et la colonne 'Volume'. 
Par conséquent, nous utilisons la fonction drop dans la bibliothèque pandas pour supprimer ces deux colonnes, ici, 
dans la commande, nous devons noter que axis = 1 (supprimer par colonne), pas par défaut à 0 car axis = 0 
(supprimer par variable)."""

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
from sklearn.preprocessing import StandardScaler

#We create function to scale data, the return is 4 terms 'Open','High','Low','Close'
#We normalize the data using scikit-learn library with StandardScaler:
def normalize_data_HHV (dementionality_reduction_HVH):
    #rescale back to -1 and 1 with 1 is highest data source and -1 is lowest data
    scaler_HHV = StandardScaler ()
    dementionality_reduction_HVH ['Open' ] = scaler_HHV.fit_transform (dementionality_reduction_HVH.Open.values.reshape (-1,1))
    dementionality_reduction_HVH ['High' ] = scaler_HHV.fit_transform (dementionality_reduction_HVH.High.values.reshape (-1,1))
    dementionality_reduction_HVH ['Low'  ] = scaler_HHV.fit_transform (dementionality_reduction_HVH.Low.values.reshape  (-1,1))
    dementionality_reduction_HVH ['Close'] = scaler_HHV.fit_transform (dementionality_reduction_HVH.Close.values.reshape(-1,1))
    return dementionality_reduction_HVH
HHV_data_exploration_normalized = normalize_data_HHV(HHV_used_data)

#NORMALIZATION FOR SFI
def normalize_data_SFI (dementionality_reduction_SFI):
    scaler_SFI = StandardScaler ()
    dementionality_reduction_SFI ['Open'] = scaler_SFI.fit_transform  (dementionality_reduction_SFI.Open.values.reshape (-1,1))
    dementionality_reduction_SFI ['High'] = scaler_SFI.fit_transform  (dementionality_reduction_SFI.High.values.reshape (-1,1))
    dementionality_reduction_SFI ['Low']  = scaler_SFI.fit_transform  (dementionality_reduction_SFI.Low.values.reshape  (-1,1))
    dementionality_reduction_SFI ['Close'] = scaler_SFI.fit_transform (dementionality_reduction_SFI.Close.values.reshape(-1,1))
    return dementionality_reduction_SFI
SFI_data_exploration_normalized = normalize_data_SFI (SFI_used_data)

#NORMALIZATION FOR PET
def normalize_data_PET (dementionality_reduction_PET):
    scaler_PET = StandardScaler ()
    dementionality_reduction_PET ['Open'] = scaler_PET.fit_transform  (dementionality_reduction_PET.Open.values.reshape (-1,1))
    dementionality_reduction_PET ['High'] = scaler_PET.fit_transform  (dementionality_reduction_PET.High.values.reshape (-1,1))
    dementionality_reduction_PET ['Low']  = scaler_PET.fit_transform  (dementionality_reduction_PET.Low.values.reshape  (-1,1))
    dementionality_reduction_PET ['Close'] = scaler_PET.fit_transform (dementionality_reduction_PET.Close.values.reshape(-1,1))
    return dementionality_reduction_PET
PET_data_exploration_normalized = normalize_data_PET (PET_used_data)

#DONNÉES SPLIT POUR HHV
#Les données fractionnées sont une étape cruciale pour déterminer la compatibilité des données prédites avec les données réelles
#Dans ce cas, nous avons choisi 70% de données brutes pour l'entraînement, et à partir de cela, nous avons prédit 30% du temps restant.
#Enfin, nous avons comparé nos données prédites avec des données réelles.

def total_data_HHV (vnstock_data_HHV, lenght_test_HHV):
    # convertir l'objet stock dataframe donné en représentation 1D Numpy-array
    vnstock_online_material_HHV = vnstock_data_HHV.values
    data_HHV = [] #1D array
    for alpha in range(len(vnstock_online_material_HHV) - lenght_test_HHV):
        data_HHV.append(vnstock_online_material_HHV [ alpha : alpha + lenght_test_HHV ])
    return data_HHV

def total_train_test_data_HHV (data_train_test_HHV , percentage_of_data_for_training_HHV = 0.7):
   print('The number of used data for the model build process',len(data_train_test_HHV))
   data_HHV = np.asarray(data_train_test_HHV,dtype=None,order='K')
   data_size_HHV = len(data_HHV)
   number_of_traning_case_HHV = int(np.floor(percentage_of_data_for_training_HHV*data_size_HHV))  #types de données est int
   x_train_HHV = data_HHV [:number_of_traning_case_HHV ,:-1,:]
   y_train_HHV = data_HHV [:number_of_traning_case_HHV ,-1 ,:]
   x_test_HHV  = data_HHV [ number_of_traning_case_HHV:,:-1,:]
   y_test_HHV  = data_HHV [ number_of_traning_case_HHV:,-1 ,:]
   return [x_train_HHV, y_train_HHV, x_test_HHV, y_test_HHV]
length_test_HHV = 5  #5vs1++
data_price_HHV = total_data_HHV (HHV_data_exploration_normalized,length_test_HHV)

x_train_HHV, y_train_HHV, x_test_HHV, y_test_HHV = total_train_test_data_HHV (data_price_HHV, 0.7)

x_train_HHV_type = type(x_test_HHV)
y_train_HHV_type = type(y_train_HHV)
x_test_HHV_type = type(x_test_HHV)
y_test_HHV_type = type(y_test_HHV)

print ('Le type de données de x_train_HHV,y_train_HHV,x_test_HHV,y_test_HHV est',
       x_train_HHV_type,y_train_HHV_type,x_test_HHV_type,y_test_HHV_type)

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
    data_SFI = np.asarray(data_train_test_SFI,dtype=None,order='K')
    data_size_SFI = len(data_SFI)
    number_of_traning_case_SFI = int(np.floor(percentage_of_data_for_training_SFI * data_size_SFI)) #types de données est int
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
   data_PET = np.asarray(data_train_test_PET,dtype=None,order='K')
   data_size_PET = len(data_PET)
   number_of_traning_case_PET = int(np.floor(percentage_of_data_for_training_PET * data_size_PET)) #types de données est int
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

#Sigmoid

x_sigmoid = np.linspace(-10, 10, 100)
y_sigmoid = 1 / (1 + np.exp(-x_sigmoid)) #Activate function
plt.plot(x_sigmoid, y_sigmoid)
plt.xlabel("x")
plt.ylabel("Sigmoid")
plt.show()

#Tanh

x_tanh = np.linspace(-np.pi, np.pi, 12)
y_tanh = np.tanh(x_tanh)
plt.plot(x_tanh, y_tanh, color='g', marker="^")
plt.title("Tanh Activate Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#La fonction mathématique de Leaky ReLU est f(x)=max(0.05x,x)
def Leaky_ReLU (x):
    return_data_per_point = [max(0.05 * epsilon, epsilon) for epsilon in x]
    return np.array(return_data_per_point, dtype=float)
x_leaky_ReLu = np.linspace (-20,20,100)
y_leaky_ReLu = Leaky_ReLU (x_leaky_ReLu)
plt.plot (x_leaky_ReLu,y_leaky_ReLu,color='g',marker="o")
plt.title ("Leaky ReLU")
plt.xlabel ("x")
plt.ylabel ("y")
plt.show ()

"""Nous construisons un modèle pour former les données pour le symbole boursier HHV, nous initialisons d'abord la classe 
Sequenial pour aider à former un cluster de classes qui sont empilées linéairement dans tf.keras.Model. Parce que 
l'équipe a vérifié le calibre auparavant, si nous n'initialisons qu'une seule couche de lSTM, un sous-ajustement se 
produira, nous construisons donc le modèle avec 2 couches de LSTM, de plus, dans le modèle, il y a de nombreuses couches 
entièrement connectées, il y a trop de paramètres, les nœuds du réseau sont trop dépendants les uns des autres pendant 
le processus d'apprentissage, ce qui va limiter l'efficacité de chaque nœud, conduisant à une sur-combinaison des nœuds 
et conduisant à un surajustement des données (over-fitting), on a donc pour créer une couche de suppression supplémentaire 
dans chaque couche LSTM pour limiter cela. Nous utilisons la fonction 'Dense' pour déclarer un calque calque pour Keras, 
avec output.dim étant le nombre de dimensions de sortie du calque lui-même, ici nous avons 4 paramètres [Open, High, Low, Close] 
donc le nombre de dimensions le la taille de sortie est de 4 et l'activation est la fonction d'activation de la couche 
[linéaire, reLU, tanh, sigmoïde], le groupe de fonctions d'activation est reLU car il correspond le mieux au modèle 
prédictif"""

#BUILDING DATA TRAINING MODEL FOR HHV
from keras.layers import LSTM

model_HHV = Sequential ()
#input_shape = [timesteps , n_features]
model_HHV.add(LSTM(units = 60 , input_shape= (x_train_HHV.shape[1],x_train_HHV.shape[1]) ,return_sequences = True))
model_HHV.add(Dropout(0.2))
model_HHV.add(LSTM(60,return_sequences = False))
model_HHV.add(Dropout(0.2))
model_HHV.add(Dense(units = 4))

"""Nous utilisons la fonction d'activation ReLu (unité linéaire rectifiée). La fonction d'activation linéaire rectifiée 
ou ReLU est une fonction non linéaire ou une fonction linéaire par morceaux qui produira l'entrée directement si elle 
est positive. Sinon, elle produira zéro. C'est la fonction d'activation la plus couramment utilisée dans les réseaux de 
neurones, en particulier dans les réseaux de neurones convolutifs (CNN) et les perceptrons multicouches. C'est simple 
mais c'est plus efficace que ses prédécesseurs comme sigmoïde ou tanh. Mathématiquement, il s'exprime par : f(x)=max(0,x)"""

model_HHV.add(Activation('relu'))

"""Pour évaluer le modèle prédictif, nous ne montrons pas seulement le graphique de corrélation entre la valeur prédite 
et la valeur réelle (parce que parfois il y a trop de données prédictives, nous ne pouvons pas tout contrôler, donc l'équipe 
utilise les mauvaises données). Squared-Error (MSE) MSE est la métrique la plus couramment utilisée pour les problèmes 
de régression dont la fonction est de trouver l'erreur quadratique moyenne entre les valeurs prédites et MSE renverra au 
programmeur un facteur qui est toujours supérieur ou égal à zéro et les valeurs aussi proche de zéro que possible 
(généralement 3 décimales)"""

#Attention : Disponible uniquement dans la version TensorFlow v2.11.0

model_HHV.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3,beta_1=0.9,beta_2=0.999,
                epsilon=1e-07,amsgrad=False,weight_decay=None,clipnorm=None,clipvalue=None,global_clipnorm=None,use_ema=False,
                ema_momentum=0.99,ema_overwrite_frequency=None,jit_compile=True,name='Adam'),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)])

model_checkpoint_callback_for_HHV = tf.keras.callbacks.ModelCheckpoint (filepath='HHV_Stock.h5',verbose=1, save_best_only=True)

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
model_SFI.add(Activation('relu'))

#Optimiseur qui implémente l'algorithme Adadelta

model_SFI.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.experimental.Adadelta(learning_rate=0.001,
                    rho=0.95,epsilon=1e-07,ema_momentum=0.99,ema_overwrite_frequency=None,jit_compile=True,
                    name='Adadelta',), metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)])

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
model_PET.add(Activation('relu'))

model_PET.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=0.01,momentum=0.0,
                nesterov=False,amsgrad=False,use_ema=False,ema_momentum=0.99,ema_overwrite_frequency=None,jit_compile=True,name='SGD'),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None)])

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

"""Dans le but d'évaluer la précision du modèle prédit, nous pouvons utiliser de nombreuses méthodes telles que RSME, 
MSE,... Dans ce cas, nous avons choisi MSE (Mean squared error). En statistique, l'erreur quadratique moyenne (MSE ) 
ou l'écart quadratique moyen (MSD) d'un estimateur. Une MSE de zéro, ce qui signifie que l'estimateur prédit les observations 
du paramètre avec une précision parfaite, est idéale (mais généralement pas possible)"""

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