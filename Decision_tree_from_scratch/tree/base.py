"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal
from typing import Union
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index, avg_gini_coefficient, reduction_in_variance


np.random.seed(42)

@dataclass
class Condition:
    '''Class to represent a condition for a decision tree node
    like the conditional operator, value to be compared with'''
    logical_operator: str
    value: Union[int, str, float, bool]

    def __init__(self,logical_operator, value ):
        self.logical_operator = logical_operator
        self.value = value  
    
    def evaluate(self,operand):
        if self.logical_operator == '==':
            return operand == self.value
        elif self.logical_operator == '!=':
            return operand != self.value
        elif self.logical_operator == '>':
            return operand > self.value
        elif self.logical_operator == '<':
            return operand < self.value
        elif self.logical_operator == '>=':
            return operand >= self.value
        elif self.logical_operator == '<=':
            return operand <= self.value
        elif self.logical_operator == 'in':
            return operand in self.value
        elif self.logical_operator == 'not in':
            return operand not in self.value
        elif self.logical_operator == 'is':
            return operand is self.value
        elif self.logical_operator == 'is not':
            return operand is not self.value
        elif self.logical_operator == 'and':
            return operand and self.value
        elif self.logical_operator == 'or':
            return operand or self.value
        elif self.logical_operator == 'not':
            return not operand

@dataclass
class DTree:
    condition: dict()  # List of conditions to be satisfied to reach the node
    CHILDREN: dict()
    Label: Union[int, float, str]    # Label contains the output class. 
    Name: str = None
    condition_type: str = None

    def isleafNode(self):
        if len(self.condition) == 0:
            return self.Label
        return None
    
    def addCondition(self, cond, NODE):
        self.condition.update(cond)
        self.CHILDREN.update(NODE)  
        

    def getChild(self, value):
        for i in range(len(self.CHILDREN)):
            if self.condition[i] == value:
                return self.CHILDREN[i]
        return None    
    
    def addLabel(self, label):
        self.Label = label



@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"] # criterion won't be used for regression
    max_depth: int = 100 # The maximum depth the tree can grow to

    root: DTree = None

    def traverse_tree(self, root):
        '''Function to traverse the tree'''
        print()
        if root.isleafNode():
            print(root.Label)
            return
        print("ROOT.NAME",root.Name)
        print(root.CHILDREN)
        for key in root.CHILDREN.keys():
            print("ROOT.KEY",key)
            self.traverse_tree(root.CHILDREN[key])

    def build_decision_tree(self, X, Y, current_depth=0):
        '''Function to build the decision tree recurively'''
        if len(Y.unique()) == 1:
            return DTree({}, {}, Y.unique()[0])
        elif len(X.columns) == 0:
            return DTree({}, {}, Y.mode()[0])
        elif self.max_depth == current_depth:
            return DTree({}, {}, Y.mode()[0])
        else:

            bestAttr, split_value = self.findBestAttr(X, Y)

            tree = DTree({}, {}, None, bestAttr)
            
            # print("bestAttr",bestAttr)
            # print('type: ',X[bestAttr].dtype)
            if X[bestAttr].dtype != 'category':


                # print(X.shape)
                # print(X[0].shape)
                # print(Y.shape)

                # plt.figure(figsize=(4, 3))
                # ax = plt.axes()
                # ax.scatter(X[0], X[1], c = Y)
                # if bestAttr == 0:
                #     ax.plot([split_value]*len(X[1]), X[bestAttr])
                # else:
                #     ax.plot(X[bestAttr], [split_value]*len(X[0]))
                # ax.set_xlabel('x[0]')
                # ax.set_ylabel('x[1]')
                # plt.show()


                X_subset_le = X[X[bestAttr] <= split_value]
                # print('Y')
                # print(Y)
                # print('X[{}]'.format(bestAttr))
                # print(X[bestAttr])
                # print(X[bestAttr] <= split_value)
                Y_subset_le = Y[X[bestAttr] <= split_value]
                X_subset_gt = X[X[bestAttr] > split_value]
                Y_subset_gt = Y[X[bestAttr] > split_value]
                
                subtree1 = self.build_decision_tree(X_subset_le, Y_subset_le, current_depth+1)
                cond1 = { (split_value,'<='): Condition('<=', split_value)}
                node = {(split_value,'<='): subtree1}
                tree.addCondition(cond1, node)
                tree.condition_type = 'real'

                subtree2 = self.build_decision_tree(X_subset_gt, Y_subset_gt, current_depth+1)
                cond2 = {(split_value,'>') : Condition('>', split_value)}
                node = {(split_value,'>'): subtree2}
                tree.addCondition(cond2, node)
                tree.condition_type = 'real'

            else:
                for value in X[bestAttr].unique():
                    X_subset = X[X[bestAttr] == value].drop(bestAttr, axis=1)
                    Y_subset = Y[X[bestAttr] == value]

                    subtree = self.build_decision_tree(X_subset, Y_subset, current_depth+1)
                    cond = {(value,'='): Condition('==', value)}
                    node = {(value, '='): subtree}
                    tree.addCondition(cond, node)
                    tree.condition_type = 'categorical'
            return tree

 
    def findBestAttr(self, X: pd.DataFrame, y: pd.Series) ->(str, float):
        
        if y.dtypes == 'object' or y.dtypes == 'category':
            if self.criterion == "information_gain":     
                bestAttr = None
                bestGain = -1* (math.log2(len(y))+1)
                split_value = None           
                for attr in X.columns:
                    gain, split = information_gain(y, X[attr])
                    if gain > bestGain:
                        bestGain = gain
                        bestAttr = attr  
                        if split:
                            split_value = split
                return bestAttr, split_value

            else:
                bestAttr = None
                bestGini = len(y)
                split_value = None
                for attr in X.columns:
                    gain, split = avg_gini_coefficient(y, X[attr])
                    if gain < bestGini:
                        bestGini = gain
                        bestAttr = attr
                        if split:
                            split_value = split
                return bestAttr, split_value
        else:
            bestAttr = None
            bestGain = -1
            split_value = None
            for attr in X.columns:
                gain, split = reduction_in_variance(y, X[attr])
                if gain > bestGain:
                    bestGain = gain
                    bestAttr = attr
                    if split:
                        split_value = split
            return bestAttr, split_value
            

 

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_copy = X.copy()
        y_copy = y.copy()
        tree = self.build_decision_tree(X_copy, y_copy)
        self.root = tree

    def compute_result(self, tree: DTree, row) -> Union[str, float, int]:
        res = tree.isleafNode()
        if res != None:
            return res

        for t in tree.condition.keys():
            ans = tree.condition[t].evaluate(row[tree.Name])
            if ans:
                return self.compute_result(tree.CHILDREN[t], row)
        return -1     

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ans = []
        for i in range(len(X)):
            ans.append(self.compute_result(self.root, X.iloc[i]))
        ans = pd.Series(ans)
        return ans

        
    def plot_print(self, k, node):
        # print(node.Label)
        
        if node.condition_type == 'categorical':
            if node.Label != None:
                print('\t'*(k), 'Y: Class {}'.format(node.Label))
                return
            for key in node.condition.keys():
                print('\t'*(k), 'Y: ', end='')
                print('?(X{0} {1} {2})'.format(node.Name, key[1], key[0]))
                self.plot_print(k+1, node.CHILDREN[key])
        else:
            if node.Label != None:
                print('\t'*(k), 'Y: Class {}'.format(node.Label))
                return
            keys = list(node.condition.keys())
            print('\t'*(k), 'Y: ', end='')
            print('?(X{0} {1} {2}'.format(node.Name, keys[1][1], keys[1][0]))
            self.plot_print(k+1, node.CHILDREN[keys[1]])
            print('\t'*(k), 'N: ', end='')
            print('?(X{0} {1} {2}'.format(node.Name, keys[0][1], keys[0][0]))
            self.plot_print(k+1, node.CHILDREN[keys[0]])


        

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.plot_print( 0, self.root)

