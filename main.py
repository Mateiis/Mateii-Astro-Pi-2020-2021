import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from colorspacious import cspace_converter
from collections import OrderedDict

font = {'fontname':'Verdana'}

"""
Because for each experiment we used 5 samples for a better area accuracy, after the experiment we took the median area, hence the "Avg_Area" in our tables
"""
Experiment1 = {
        'Temperature': [28.21,28.21,28.21,28.21,28.21,28.21,28.21,28.21,28.21,28.21,28.21],
        'Humidity': [46.83,46.83,46.83,46.83,46.83,46.83,46.83,46.83,46.83,46.83,46.83],
        'Time': [5,16,21,34.5,45,55.5,71,77.5,84.5,97.5,108.5],
        'Avg_Area': [0.3,0.62,2.18,4.13,6.58,14.54,24.76,34.23,37.7,44.62,53.84],
        } # Slope is: 1.86846824


Experiment2 = {
        'Temperature': [18.5,18.5,18.5,18.5,18.5],
        'Humidity': [59,59,59,59,59],
        'Time': [5,25.5,37,44,60],
        'Avg_Area': [0.39,0.46,0.68,1.9,3.2]
        } # Slope is: 0.71874192


Experiment3 = {
        'Temperature': [30,30,30,30,30,30],
        'Humidity': [92.31,92.31,92.31,92.31,92.31,92.31],
        'Time': [14.5,19,23,29,34.5,42],
        'Avg_Area': [0.39,0.54,3.51,8.11,18.18,53.35]
        } # Slope is: 4.86631424

Experiment4 = {
        'Temperature': [27.5,27.5,27.5,27.5,27.5,27.5,27.5],
        'Humidity': [42.5,42.5,42.5,42.5,42.5,42.5,42.5],
        'Time': [5,15.5,22.5,45,58,69,81.5],
        'Avg_Area': [0.273,0.5,1.5,6.41,12.37,23.28,36.19]
        } # Slope is: 1.84023553

Experiment5 = {
        'Temperature': [31.2,31.2,31.2,31.2,31.2],
        'Humidity': [57.3,57.3,57.3,57.3,57.3],
        'Time': [5,15.5,22.5,45,58],
        'Avg_Area': [0.2058,0.929,5.762,29.413,53.398]
        } # Slope is: 2.35425092


Experiment6 = {
        'Temperature': [21.8,21.8,21.8,21.8,21.8],
        'Humidity': [81.66,81.66,81.66,81.66,81.66],
        'Time': [5,24,39,46.5,50],
        'Avg_Area': [0.574,5.807,14.9,20.686,29.224]
        } # Slope is: 1.64779042

Experiment7 = {
        'Temperature': [30,30,30,30,30,30],
        'Humidity': [33.5,33.5,33.5,33.5,33.5,33.5],
        'Time': [5,14.5,23,45,57,69.5],
        'Avg_Area': [0.291,1.11,2.463,7.379,13.756,20.20]
        } # Slope is: 1.6177985



"""
Creating data frames in order to find the slope and coefficient for our experiments
"""

df1 = pd.DataFrame(Experiment1,columns=['Time','Avg_Area'])
df2 = pd.DataFrame(Experiment2,columns=['Time','Avg_Area'])
df3 = pd.DataFrame(Experiment3,columns=['Time','Avg_Area'])
df4 = pd.DataFrame(Experiment4,columns=['Time','Avg_Area'])
df5 = pd.DataFrame(Experiment5,columns=['Time','Avg_Area'])
df6 = pd.DataFrame(Experiment6,columns=['Time','Avg_Area'])
df7 = pd.DataFrame(Experiment7,columns=['Time','Avg_Area'])


X1 = df1[['Time']]
Y1 = df1['Avg_Area']

X2 = df2[['Time']]
Y2 = df2['Avg_Area']

X3 = df3[['Time']]
Y3 = df3['Avg_Area']

X4 = df4[['Time']]
Y4 = df4['Avg_Area']

X5 = df5[['Time']]
Y5 = df5['Avg_Area']


X6 = df6[['Time']]
Y6 = df6['Avg_Area']

X7 = df7[['Time']]
Y7 = df7['Avg_Area']

lr1 = LinearRegression()
lr2 = LinearRegression()
lr3 = LinearRegression()
lr4 = LinearRegression()
lr5 = LinearRegression()
lr6 = LinearRegression()
lr7 = LinearRegression()
"""
Applying ln to the equation of our mould growth (Area = Coef * (Time ^ Slope) ) to transform it from a power function to a linear one
and then finding the slope and Y-Intercept ( ln(Coef) which we will later transform back into Coef by raising it to the power of e)
"""
lr1.fit(np.log(X1),np.log(Y1))
lr2.fit(np.log(X2),np.log(Y2))
lr3.fit(np.log(X3),np.log(Y3))
lr4.fit(np.log(X4),np.log(Y4))
lr5.fit(np.log(X5),np.log(Y5))
lr6.fit(np.log(X6),np.log(Y6))
lr7.fit(np.log(X7),np.log(Y7))

Slope1 = lr1.coef_
Slope2 = lr2.coef_
Slope3 = lr3.coef_
Slope4 = lr4.coef_
Slope5 = lr5.coef_
Slope6 = lr6.coef_
Slope7 = lr7.coef_

#The coeffiecients are
Coef1 = lr1.intercept_
Coef2 = lr2.intercept_
Coef3 = lr3.intercept_
Coef4 = lr4.intercept_
Coef5 = lr5.intercept_
Coef6 = lr6.intercept_
Coef7 = lr7.intercept_


"""
Building the final table which will allow us to find the linear depdence between the slope, humidity and temperature.
It will also find the dependence between the slope and Y-Intercept, which we have also found to be linear.
"""
Final_Table = {
        'Temperature': [28.21,18.5,30,27.5,31.2,21.8,30],
        'Humidity': [46.83,59,92.31,42.5,57.3,81.66,33.5],
        'Slope': [Slope1[0],Slope2[0],Slope3[0],Slope4[0],Slope5[0],Slope6[0],Slope7[0]],
        'Y-Intercept': [Coef1,Coef2,Coef3,Coef4,Coef5,Coef6,Coef7]
        }

CoefDataFrame = pd.DataFrame(Final_Table,columns=['Slope','Y-Intercept'])

Xc = CoefDataFrame[['Slope']]
Yc = CoefDataFrame['Y-Intercept']

Coefficient_Regression = LinearRegression()
Coefficient_Regression.fit(Xc,Yc)

dff = pd.DataFrame(Final_Table,columns=['Temperature','Humidity','Slope'])

Xf = dff[['Temperature','Humidity']]
Yf = dff['Slope']

lrf = LinearRegression()
lrf.fit(Xf,Yf)

"""
Creating Area vs Time graph for any given temperature and humidity
"""
New_Temperature = 28
New_Humidity = 47
Predicted_Slope = lrf.predict([[New_Temperature ,New_Humidity]])[0]
ln_Predicted_Coefficient = Coefficient_Regression.predict([[Predicted_Slope]])[0] #Transforming the ln(Coef) into Coef
Predicted_Coefficient = math.exp(ln_Predicted_Coefficient)
print ('Predicted Growth Slope: \n', lrf.predict([[New_Temperature ,New_Humidity]]))
print ('Predicted Coefficient: \n', Predicted_Coefficient)

x = np.linspace(1,72,100)  #Simulating the first 72 hours
y = Predicted_Coefficient*(x**Predicted_Slope)  #Writing the equation for Area with respect to time
plt.plot (x,y, color = '#C11D4B')
plt.xlabel("Time (Hours)")
plt.ylabel("Area (cm^2)")
plt.title("T = 28°C     Hum = 47%",fontsize = 16,  color='#269C82', **font, weight = 'bold')
plt.show()




"""
Generating for each humidity from 30% to 60% its corresponding temperature values so that the GI (Growth Index) is minimal.
This was used for generating the table in the 'Conclusion' section
"""

'''
Kantor = 0
Hum_Temp = 0
minGI = 100
Sum=0

for New_Humidity in np.arange (30,60,1):
    minIndex = 101
    for New_Temperature in np.arange (17,30,0.5):
        Sum=0
        Predicted_Slope = lrf.predict([[New_Temperature ,New_Humidity]])[0]
        ln_Predicted_Coefficient = Coefficient_Regression.predict([[Predicted_Slope]])[0]
        Predicted_Coefficient = math.exp(ln_Predicted_Coefficient)
        if(Predicted_Slope > 0): #To ensure that our results predict physically possible behaviour
            #Area = Predicted_Coefficient*(72**Predicted_Slope) #Area after 72 hours
            for i in range (1,71,1):
                Growth_Rate_x = (Predicted_Coefficient*((i+1)**Predicted_Slope)-Predicted_Coefficient*(i**Predicted_Slope))/(Predicted_Coefficient*(i**Predicted_Slope))
                Sum = Sum + Growth_Rate_x
            GI = Sum/71 * 100
            if(minGI > GI):
                minGI = GI
                Hum_Temp = New_Temperature
                ShownSlope = Predicted_Slope
                ShownCoefficient = Predicted_Coefficient
    Kantor = Kantor + 1
    print("Data set number: ", Kantor)
    print ("Temperature is: ",Hum_Temp)
    print("Humidity is: ",New_Humidity)
    print("Slope is: ",ShownSlope)
    print("Coefficient is: ",ShownCoefficient)
    print("Average Growth Rate: ", minGI)
    print("              ")
'''






"""
This section in the code plots experimental data compared to predicted values, in orded to test de accuracy of the model
"""

'''
New_Temperature = 30
New_Humidity = 33.5
Predicted_Slope = lrf.predict([[New_Temperature ,New_Humidity]])[0]
ln_Predicted_Coefficient = Coefficient_Regression.predict([[Predicted_Slope]])[0] #Transforming the ln(Coef) into Coef
Predicted_Coefficient = math.exp(ln_Predicted_Coefficient)
print(Predicted_Coefficient)
print(Predicted_Slope)

x = np.linspace(1,72,130)
y1 = math.exp(lr7.intercept_)*(x**Slope7[0])
y2 = Predicted_Coefficient*(x**Predicted_Slope)
plt.xlabel("Time (Hours)")
plt.ylabel("Area (cm^2)")
plt.plot(x, y1, color = "#C11D4B" , label = "ISS'S Mould Growth Over 72 Hours")
plt.plot(x, y2, color = "#269C82", label = "Model's Predictions")
plt.suptitle("Conditions On The ISS", fontsize = 20, color='#269C82', **font, weight = 'bold', y = 1)
plt.title("T = 30°C     Hum = 33.5%",fontsize = 16,  color='#269C82', **font, weight = 'bold')
plt.legend()
plt.show()
'''




"""
Generating the 4D Graph
"""


def hex_to_rgb(value):

    '''Converts hex to rgb colours'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):

    '''Converts rgb to decimal colours'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):

    ''' creates and returns a color map that can be used in heat map figures'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


''' Generating the colors for our gradient'''
hex_list = ['#269c82','#309a77','#3a986d','#449562','#4d9256','#568f4c','#5f8c41','#678937','#70852d','#788123','#807c1a','#887712','#90720c','#976c09','#9f660a','#a55f0f','#ac5716','#b24f1d','#b74725','#be3437','#c02941','#c11d4b']

Te = []
Hu = []
Ti = []
Ar = []

for New_Temperature in range (15,37,1):
    for New_Humidity in np.arange (20,60,1):
        Predicted_Slope = lrf.predict([[New_Temperature ,New_Humidity]])[0]
        if(Predicted_Slope>0):
            ln_Predicted_Coefficient = Coefficient_Regression.predict([[Predicted_Slope]])[0]
            Predicted_Coefficient = math.exp(ln_Predicted_Coefficient)
            for Time in range(1,72,1):
                Area = Predicted_Coefficient*(Time**Predicted_Slope)
                Te.append(New_Temperature)
                Hu.append(New_Humidity)
                Ti.append(Time)
                Ar.append(Area)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = Te
y = Hu
z = Ti
c = Ar
img = ax.scatter(x, y, z, c=c, cmap=get_continuous_cmap(hex_list,float_list = [0, 0.005, 0.007, 0.009, 0.012, 0.015,0.026, 0.032, 0.038, 0.044, 0.05, 0.06, 0.07, 0.1,0.15,0.3,0.4,0.6,0.7,0.8,0.9,1]))
fig.colorbar(img).set_label("Area (cm^2)", **font)
ax.set_xlabel("Temperature (°C)", **font)
ax.set_ylabel("Humidity (%)", **font)
ax.set_zlabel("Time (Hours)", rotation = -90, **font)
ax.set_title("4D Graph - Mould Growth Model", color = '#269c82', weight = 'bold',**font)
plt.show()
