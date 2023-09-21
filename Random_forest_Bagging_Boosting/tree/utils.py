import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def entropy(Y: pd.Series, wt: pd.Series = None) -> float:

    # class_counter = dict() 
    # for y in Y:
    #     class_counter[y] = class_counter.get(y,0) + 1

    # input()
    if wt is None:
        wt = pd.Series([1]*len(Y))

    class_counter = Y.value_counts()
    total_weight = wt.sum()

    df = pd.DataFrame({'y':Y, 'w': wt})
    entropy = 0
    # print("===========================")
    # print(df)
    # print("class_counter",class_counter)
    for y in class_counter.index:
        if class_counter[y] == 0:
            continue
        # print("y",y)
        p = df['w'][df['y'] == y].sum()/total_weight
        # print("Total Weight",total_weight)
        # print("df['w'][df['y'] == y]",df['w'][df['y'] == y])
        # if p == 0:
        #     print("wt ",wt)
        #     print('Y',Y)
        #     input()
        # print("--------------")
        entropy -= p * np.log2(p)
        # input()

    return entropy

def gini_index(Y: pd.Series, sample_wieghts: pd.Series = None) -> float:
    """
    Function to calculate the gini index
    """
    if len(Y) == 0:
        return 0
    
    if sample_wieghts is None:
        sample_wieghts = pd.Series([1]*len(Y))
        
    total_weights = sample_wieghts.sum()

    df = pd.DataFrame({'y':Y, 'w': sample_wieghts})
    class_counter = df.groupby('y').sum()
    gini = 1 - (class_counter['w']**2).sum()/(total_weights**2)

    return gini

def avg_gini_coefficient(Y: pd.Series, X: pd.Series, wt: pd.Series = None) -> (float,float):
    """
    Function to calculate the average gini coefficient
    """
    if len(Y) == 0:
        return (0, None)

    if wt is None:
        wt = pd.Series([1]*len(Y))
    
    #--- discrete input - discrete output -----------------------
    if X.dtype == "object" or X.dtype == "category":
        X = X.astype('category')
        # create the dictionary of each input category
        class_count = X.value_counts()

        weighted_gini = 0
        total_weight = wt.sum()

        for c in class_count.index:
            # calculate the gini index of the class
            gini_index_of_Xc = gini_index(Y[X == c], wt[X == c] ) 
            # calculate the weight of the class
            class_weight = wt[X == c].sum()
            # calculate the weighted gini index
            weighted_gini += (class_weight/total_weight) * gini_index_of_Xc
        return (weighted_gini, None)      

    #--- continuous input - discrete output ---------------------
    else:
        X = X.astype('float')
        # TODO sort Y and attr by attr 
        df = pd.DataFrame({'attr': X, 'Y': Y, 'wt': wt})
        df = df.sort_values(by=['attr'])

        if len(Y) == 1:
            return (0,None)

        min_gini = 1
        split_value = df['attr'].iloc[0]
        gini_of_y = gini_index(df['Y'], df['wt'])

        for y in range(len(Y)-1):
            
            # if df['Y'].iloc[y] == df['Y'].iloc[y+1]:
            #     continue
            if df['attr'].iloc[y] == df['attr'].iloc[y+1]:
                continue
            
            split = (df['attr'].iloc[y] + df['attr'].iloc[y+1])/2
            
            # Calculate the gini index of the left and right splits
            gi_of_lt_split = gini_index(df['Y'][df['attr'] <= split], df['wt'][df['attr'] <= split])
            gi_of_gt_split = gini_index(df['Y'][df['attr'] > split], df['wt'][df['attr'] > split])

            # Calculate the weights
            class_wieghts_ls_split = df['wt'][df['attr'] <= split].sum()
            class_wieghts_gt_split = df['wt'][df['attr'] > split].sum()
            total_weight = df['wt'].sum() 

            weighted_gini = 0
            weighted_gini += (class_wieghts_ls_split/total_weight * gi_of_lt_split)
            weighted_gini += (class_wieghts_gt_split/total_weight * gi_of_gt_split)

            # findig the minimum gini index
            if weighted_gini < min_gini:
                min_gini = weighted_gini
                split_value = split
            
        return (min_gini , split_value)



def information_gain(Y: pd.Series, attr: pd.Series, wt: pd.Series = None) -> (float, float):
    """
    Function to calculate the information gain
    """
    if len(Y) == 0:
        return 0, None
    if wt is None:
        wt = pd.Series([1]*len(Y))

    total_weight = wt.sum()

    #--------------Discrete input - discrete output--------------------------
    if attr.dtype == "object" or attr.dtype == "category":
        attr = attr.astype('category')
        # attr = attr.cat.codes
        class_count = attr.value_counts()
        gain = entropy(Y, wt)
        # total_weight = df['wt'].sum()
        for c in class_count.index:
            class_weight = wt[attr == c].sum()
            gain -= (class_weight/total_weight) * entropy(Y[attr == c], wt[attr == c])

        return (gain, None)
    
    #------------------ Continous input - discrete output ------------------
    else:
        attr = attr.astype('float')
        # gain = entropy(Y)

        # TODO sort Y and attr by attr 
        df = pd.DataFrame({'attr': attr, 'Y': Y, 'wt': wt})
        df = df.sort_values(by=['attr'])

        # df['attr'].astype('float')
        if len(Y) == 1:
            return 0, None

        max_gain = 0
        split_value = df['attr'].iloc[0]
        gain_of_y = entropy(df['Y'], df['wt'])

        for y in range(len(Y)-1):
            # if df['Y'].iloc[y] == df['Y'].iloc[y+1]:
            #     continue
            if df['attr'].iloc[y] == df['attr'].iloc[y+1]:
                continue
            split = (df['attr'].iloc[y] + df['attr'].iloc[y+1])/2

            # entropy of the left and right splits
            entropy_lt_split = entropy(df['Y'][df['attr'] <= split], df['wt'][df['attr'] <= split])
            entropy_gt_split = entropy(df['Y'][df['attr'] > split], df['wt'][df['attr'] > split])

            # Calculate the weights
            class_wieghts_ls_split = df['wt'][df['attr'] <= split].sum()
            class_wieghts_gt_split = df['wt'][df['attr'] > split].sum()

            gain = gain_of_y
            gain -= (class_wieghts_ls_split/total_weight) * entropy_lt_split
            gain -= (class_wieghts_gt_split/total_weight) * entropy_gt_split

            if gain > max_gain:
                max_gain = gain
                split_value = split
        return max_gain , split_value
        
        # for c in attr.unique():
        #     gain -= (len(Y[attr == c])/len(Y)) * entropy(Y[attr == c])
        # return gain


def reduction_in_variance(Y: pd.Series, attr: pd.Series, wt: pd.Series = None) -> (float, float):

    if len(Y) <= 1:
        return 0, None
    if wt is None:
        wt = pd.Series([1]*len(Y))
    total_weight = wt.sum()

    # For discrete input data - continuous output data
    if attr.dtype == "object" or attr.dtype == "category":
        attr = attr.astype('category')
        class_count = attr.value_counts()
        gain = np.var(Y)
        for c in class_count.index:
            y_col = Y[attr == c]
            if len(y_col) == 0:
                continue
            class_weight = wt[attr == c].sum()

            gain -= (class_weight/total_weight) * np.var(y_col)
        return gain, None
    
    # For continuous input data - continuous output data
    
    else:
        # TODO sort Y and attr by attr 
        df = pd.DataFrame({'attr': attr, 'Y': Y, 'wt': wt})
        df = df.sort_values(by=['attr'])
        df['attr'].astype('float')

        split_value = df['attr'].iloc[0]
        if len(Y) == 1:
            return 0, split_value

        max_gain = 0
        gain_of_y = np.var(df['Y'])
        for y in range(len(Y)-1):
            # if df['Y'].iloc[y] == df['Y'].iloc[y+1]:
            #     continue
            if df['attr'].iloc[y] == df['attr'].iloc[y+1]:
                continue
            split = (df['attr'].iloc[y] + df['attr'].iloc[y+1])/2
            # variance of the left and right splits
            variance_lt_split = np.var(df['Y'][df['attr'] <= split])
            variance_gt_split = np.var(df['Y'][df['attr'] > split])

            # Calculate the weights
            class_wieghts_ls_split = df['wt'][df['attr'] <= split].sum()
            class_wieghts_gt_split = df['wt'][df['attr'] > split].sum()


            gain = gain_of_y 
            gain -= (class_wieghts_ls_split/total_weight) * variance_lt_split
            gain -= (class_wieghts_gt_split/total_weight) * variance_gt_split
            
            if gain > max_gain:
                max_gain = gain
                split_value = split
        return max_gain , split_value




def nested_cross_validation(X, y, criteria, get_scores_func, outerFolds, innerFolds, depth_level, depth_step, check_rmse=False):
    foldX = [X[int(i/outerFolds *len(X)) : int((i+1)/outerFolds *len(X))] for i in range(outerFolds)]
    foldy = [y[int(i/outerFolds *len(y)) : int((i+1)/outerFolds *len(y))] for i in range(outerFolds)]
    testing_score = []
    for outerFold in range(1, outerFolds+1):
        testX = pd.DataFrame(foldX[outerFold-1])
        testY = pd.Series(foldy[outerFold-1])

        trainX = pd.concat([pd.DataFrame(foldX[(outerFold+i-1)%outerFolds]) for i in range(1,outerFolds)]).reset_index(drop=True)
        trainY = pd.concat([pd.Series(foldy[(outerFold+i-1)%outerFolds]) for i in range(1,outerFolds) ]).reset_index(drop=True)

        foldX_inner = [trainX[int(i/innerFolds *len(trainX)) : int((i+1)/innerFolds *len(trainX))] for i in range(innerFolds)]
        foldy_inner = [trainY[int(i/innerFolds *len(trainY)) : int((i+1)/innerFolds *len(trainY))] for i in range(innerFolds)]

        depth_accuracy_map = dict()
        for depth in range(0,depth_level+1, depth_step):
            validation_score = []
            for innerfold in range(1, innerFolds+1):
                valdX = pd.DataFrame(foldX_inner[innerfold-1])
                valdY = pd.Series(foldy_inner[innerfold-1])

                trainX_inner = pd.concat([pd.DataFrame(foldX_inner[(innerfold+i-1)%innerFolds]) for i in range(1,innerFolds)]).reset_index(drop=True)
                trainY_inner = pd.concat([pd.Series(foldy_inner[(innerfold+i-1)%innerFolds]) for i in range(1,innerFolds) ]).reset_index(drop=True)
                
                # print('trainX_inner.shape, trainY_inner.shape, valdX.shape, valdY.shape')
                # print(trainX_inner.shape, trainY_inner.shape, valdX.shape, valdY.shape)
                validation_score.append(get_scores_func(trainX_inner, trainY_inner, valdX, valdY, depth, criteria))

            depth_accuracy_map[depth] = np.mean(validation_score)
        
        print(depth_accuracy_map)
        if check_rmse:
            max_depth_accuracy = min(depth_accuracy_map, key= lambda x: depth_accuracy_map[x])
        else:
            max_depth_accuracy = max(depth_accuracy_map, key= lambda x: depth_accuracy_map[x])
        
        print("Max_depth for fold {0} is {1}".format(outerFold, max_depth_accuracy))
        plt.plot(list(depth_accuracy_map.keys()), list(depth_accuracy_map.values()))
        plt.xlabel("Depth")
        if check_rmse:
            plt.ylabel("RMSE")
            plt.title("Accuracy vs Depth for fold {0}".format(outerFold))
            plt.savefig('./imgs/Q3/rmse_d_fold{}.png'.format(outerFold))

        else:
            plt.ylabel("Accuracy")
            plt.title("Accuracy vs Depth for fold {0}".format(outerFold))
            plt.savefig('./imgs/Q3/acc_d_fold{}.png'.format(outerFold))
        # plt.legend()
        # plt.show()

        testing_score.append((max_depth_accuracy, get_scores_func(trainX, trainY, testX, testY, max_depth_accuracy, criteria)))
        print('Testing Score',testing_score[-1])
    return testing_score