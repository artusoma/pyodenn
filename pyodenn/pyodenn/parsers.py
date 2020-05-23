## Machine learning modules
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU
from keras import backend as K
from keras import optimizers

## Normal Modules
import matplotlib.pyplot as plt
import numpy as np
import re
import sys 

class Parser(object):
    '''The parser converts a string expression to a computable 
    form using the to_expression method. 
    '''
    @staticmethod
    def to_expression(to_parse, x):
        neg = False        # Keeps track if term is negative
        term = []    # Array that will be multiplied together
        expression = [] # Complete expression
        curr_num = ''      
        chars = []
        i = 0           # Incrementer for each character in a term
        
        ## Split to_parse over addition/subtraction
        to_parse = to_parse.split('+')
        
        for t in to_parse:  
            chars = list(t)
            i = 0

            while (i < len(chars)):
                if chars[i] == '-':
                    term.append(1)
                    neg = True
                    i += 1

                elif chars[i] == 'x':
                    if curr_num != '':
                        term.append(int(curr_num))
                        curr_num = ''
                    term.append(x)
                    i += 1

                elif chars[i] == '^':
                    last_term = term[-1]
                    for p in range(int(chars[i+1])-1):
                        term.append(last_term)
                    i += 2

                else:
                    curr_num += chars[i]
                    try: 
                        chars[i+1]
                    except IndexError:
                        term.append(int(curr_num))

                    i += 1
                    
            if neg:
                expression.append(-np.prod(np.array(term)))
            else:
                expression.append(np.prod(np.array(term)))

            neg = False
            term = []

        return np.array(expression).sum()