
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from tree.base import DecisionTree
from metrics import *
import seaborn as sns

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions



def plot_time(fit_stats, predict_stats, my_model_acc_stats, sklearn_acc_stats):
    '''
    Function to plot the following results: 
     - Heatmap of fit time for my DT vs sklearn DT
     - Heatmap of predict time for my DT vs sklearn DT
     - Fit time plot for my DT vs sklearn DT
     - Predict time plot for my DT vs sklearn DT
    '''

    N_list = np.array([i[0] for i in fit_stats.keys()])
    P_list = np.array([i[1] for i in fit_stats.keys()])

    my_fit_time = np.array([i[0] for i in fit_stats.values()])
    sklearn_fit_time = np.array([i[1] for i in fit_stats.values()])
    my_predict_time = np.array([i[0] for i in predict_stats.values()])
    sklearn_predict_time = np.array([i[1] for i in predict_stats.values()])

    # my_model_acc = np.array([i for i in my_model_acc_stats.values()])
    # sklearn_acc = np.array([i for i in sklearn_acc_stats.values()])
    

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(N_list, P_list, my_model_acc, label="my model")
    # ax.plot(N_list, P_list, sklearn_acc, label="sklearn")
    # ax.set_xlabel("SAMPLES")
    # ax.set_ylabel("FETURES")
    # ax.set_zlabel("Accuracy")
    # plt.title('Accuracy PLOT MY-MODEL v/s SKLEARN')
    # plt.legend()
    # plt.show()


    fit_with_time = {'num_of_samples': N_list,
                     'num_of_features': P_list,
                     'my_fit_time': my_fit_time}

    fit_with_time = pd.DataFrame(fit_with_time)
    fit_with_time = fit_with_time.pivot("num_of_samples", "num_of_features", "my_fit_time")
    sns.heatmap(fit_with_time, annot=True, fmt=".2f")
    plt.title('HEATMAP OF FIT TIME')
    plt.legend()
    # plt.savefig("./imgs/Q4/{}.png".format("hm_fit_"))
    plt.show()

    predict_with_time = {'num_of_samples': N_list,
                            'num_of_features': P_list,
                            'my_predict_time': my_predict_time}
    predict_with_time = pd.DataFrame(predict_with_time)
    predict_with_time = predict_with_time.pivot("num_of_samples", "num_of_features", "my_predict_time")
    sns.heatmap(predict_with_time, annot=True, fmt=".2f")
    plt.title('HEATMAP OF PREDICT TIME')
    plt.legend()
    plt.show()


    print("MSE- fit", np.mean((my_fit_time - sklearn_fit_time)**2))
    print("MSE- predict", np.mean((my_predict_time - sklearn_predict_time)**2))
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(N_list, P_list, my_fit_time, label="my fit")
    ax.plot(N_list, P_list, sklearn_fit_time, label="sklearn fit")
    ax.set_xlabel("SAMPLES")
    ax.set_ylabel("FETURES")
    ax.set_zlabel("TIME")
    plt.title('FIT TIME PLOT MY-MODEL v/s SKLEARN')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(N_list, P_list, my_predict_time, label="my predict")
    ax.plot(N_list, P_list, sklearn_predict_time, label="sklearn predict")
    ax.set_xlabel("SAMPLES")
    ax.set_ylabel("FETURES")
    ax.set_zlabel("TIME")
    plt.title('PREDICT TIME PLOT MY-MODEL v/s SKLEARN')
    plt.legend()
    plt.show()




max_row=100
max_col=20


def return_stats(my_model, sklearn_model, X, y):
    result = {
        'fit_time' : None,
        'predict_time' : None,
        'my_model_accuracy' : 0,
        'sklearn_model_accuracy' : 0
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #----------------- FIT MY-DT ----------------
    start = time.time()
    my_model.fit(X_train, y_train)
    end = time.time()
    net_time_to_fit_my_model = end-start
    #----------------- FIT SKLEARN-DT ----------------
    start = time.time()
    sklearn_model.fit(X_train,y_train)
    end = time.time()
    net_time_to_fit_sklearn_model = end - start


    result['fit_time'] = (net_time_to_fit_my_model, net_time_to_fit_sklearn_model)

    #----------------- PREDICT MY-DT ----------------
    start = time.time()
    y_hat_my_model = my_model.predict(X_test)
    end = time.time()
    net_time_to_predict_my_model = end-start

    #----------------- PREDICT SKLEARN-DT ----------------
    start = time.time()
    y_hat_sklearn = sklearn_model.predict(X_test)
    end = time.time()
    net_time_to_predict_sklearn_model = end - start

    result['predict_time'] = (net_time_to_predict_my_model, net_time_to_predict_sklearn_model)
    # y_hat_my_model.reset_index(drop = True, inplace = True)
    # y_test.reset_index(drop = True, inplace = True)
    # print(type(y_hat_my_model), type(y_hat_sklearn))
    # y_hat_sklearn.reset_index(drop = True, inplace = True)
    # print(y_hat_my_model)
    # print(y_test)
    # result['my_model_accuracy'] = accuracy( y_hat_my_model, y_test)
    # result['sklearn_model_accuracy'] = accuracy( pd.Series(y_hat_sklearn), y_test)

    return result



fit_stats, predict_stats = dict(), dict()
my_model_accuracy_stats = dict()
sklearn_accuracy_stats = dict()

# for real input, real output
for N in range(5,max_row,5):
    for P in range(5,max_col,5):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
        my_model = DecisionTree(criterion='information_gain', max_depth=10) #Split based on Inf. Gain
        sklearn_model=DecisionTreeRegressor(max_depth=10)
        
        result = return_stats(my_model, sklearn_model, X, y)

        fit_stats[(N,P)] = result['fit_time']
        predict_stats[(N,P)] = result['predict_time']
        my_model_accuracy_stats[(N,P)] = result['my_model_accuracy']
        sklearn_accuracy_stats[(N,P)] = result['sklearn_model_accuracy']

        # fit_t, predict_t = return_stats(my_model, sklearn_model, X, y)
        # fit_stats[(N,P)] = fit_t
        # predict_stats[(N,P)] = predict_t

print("for real input and  real output")
# plot(x1,y1,z1,z2,rmse1,rmse2)
plot_time(fit_stats, predict_stats, my_model_accuracy_stats, sklearn_accuracy_stats)
##########################################################################################################


# for discrete input and  discrete output
fit_stats, predict_stats = dict(), dict()
my_model_accuracy_stats = dict()
sklearn_accuracy_stats = dict()

for N in range(5,max_row,5):
    for P in range(5,max_col,5):
        X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
        my_model = DecisionTree(criterion='information_gain', max_depth=10) #Split based on Inf. Gain
        sklearn_model=DecisionTreeClassifier(criterion="entropy", max_depth=10)
        
        result = return_stats(my_model, sklearn_model, X, y)

        fit_stats[(N,P)] = result['fit_time']
        predict_stats[(N,P)] = result['predict_time']
        my_model_accuracy_stats[(N,P)] = result['my_model_accuracy']
        sklearn_accuracy_stats[(N,P)] = result['sklearn_model_accuracy']

print("for discrete input and  discrete output")
plot_time(fit_stats, predict_stats, my_model_accuracy_stats, sklearn_accuracy_stats)
###########################################################################################################

# for discrete input and  real output
fit_stats, predict_stats = dict(), dict()
my_model_accuracy_stats = dict()
sklearn_accuracy_stats = dict()

for N in range(5,max_row,5):
    for P in range(5,max_col,5):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
        my_model = DecisionTree(criterion='information_gain', max_depth=10)
        sklearn_model = DecisionTreeRegressor(max_depth=10)
        
        result = return_stats(my_model, sklearn_model, X, y)

        fit_stats[(N,P)] = result['fit_time']
        predict_stats[(N,P)] = result['predict_time']
        my_model_accuracy_stats[(N,P)] = result['my_model_accuracy']
        sklearn_accuracy_stats[(N,P)] = result['sklearn_model_accuracy']

print("for discrete input and  real output")
plot_time(fit_stats, predict_stats, my_model_accuracy_stats, sklearn_accuracy_stats)

###########################################################################################################


# for real input and  discrete output
fit_stats, predict_stats = dict(), dict()
my_model_accuracy_stats = dict()
sklearn_accuracy_stats = dict()

for N in range(5,max_row,5):
    for P in range(5,max_col,5):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
        my_model = DecisionTree(criterion='information_gain', max_depth=10)
        sklearn_model = DecisionTreeClassifier(criterion="entropy", max_depth=10)

        result = return_stats(my_model, sklearn_model, X, y)

        fit_stats[(N,P)] = result['fit_time']
        predict_stats[(N,P)] = result['predict_time']
        my_model_accuracy_stats[(N,P)] = result['my_model_accuracy']
        sklearn_accuracy_stats[(N,P)] = result['sklearn_model_accuracy']

print("for real input and  discrete output")
plot_time(fit_stats, predict_stats, my_model_accuracy_stats, sklearn_accuracy_stats)
#########################################################################################################


'''
 for real input and  real output    
MSE- fit 1.5902306570275817
MSE- predict 3.8264136944858534e-05

#MSE- fit 2.220127586399862
#MSE- predict 1.0232395896781344e-05

for discrete input and  discrete output
MSE- fit 0.04896488926785499
MSE- predict 6.145693279054117e-05

#MSE- fit 0.06099133065918783
#MSE- predict 4.30961987755533e-05

for discrete input and  real output
MSE- fit 0.9762117305435696
MSE- predict 2.930559947521437e-05

#MSE- fit 1.3824897744546674
#MSE- predict 4.077920215195263e-05

for real input and  discrete output
MSE- fit 0.6745632185078105
MSE- predict 2.0856170160051852e-05

#MSE- fit 2.3963049542247115
#MSE- predict 1.4879008088226126e-05
'''