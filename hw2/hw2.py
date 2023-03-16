import sys
import math
import string
import numpy as np


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    with open (filename,encoding='utf-8') as f:
        # stringtify data and make it a list of characters
        data = f.read() 
        data = list(data)

        # initialize dictionary with all characters and set it to count 0
        alphas = list(string.ascii_uppercase)
        for alpha in alphas:
            X[alpha] = 0

        # count appearance of character
        for char in data:
            char = char.upper()
            if char.isalpha():
                if char in X:
                    X[char] = X[char] + 1
    f.close()
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def main():
    filename = "letter.txt" # default file name to use

    #Q1
    X = shred(filename)
    print("Q1")
    for key in sorted(X.keys()):
        print("{0} {1}".format(key, X[key]))
    #Q1 done

    #Q2
    # get the e and s vectors
    e_s = get_parameter_vectors()
    e1 = (e_s[0][0])
    s1 = (e_s[1][0])
    # get the X vector by extract it from the X dict
    x_v = ([X[key] for key in sorted(X.keys())])
    x_1 = x_v[0]    
    
    # calculate e1 and s1 result
    result_e1 = x_1 * math.log(e1)
    result_s1 = x_1 * math.log(s1)
    
    print("Q2")
    print("%.4f" %result_e1)
    print("%.4f" %result_s1)
    # Q2 done

    #Q3
    # F(English) and F(Spanish)
    
    # e and s vector
    e = (e_s[0])
    s = (e_s[1])

    # get the doc product, do this by loop through e and multiply value with the e and s vector
    # simulating a 1 x n (X) and n x 1(e) matrix multiplication
    result_e = 0
    result_s = 0
    for i in range(len(x_v)):
        result_e += x_v[i] * math.log(e[i])
        result_s += x_v[i] * math.log(s[i])
    
    fEnglish = math.log(0.6) + result_e
    fSpanish = math.log(0.4) + result_s

    print("Q3")
    print("%.4f" %fEnglish)
    print("%.4f" %fSpanish)

    #Q4
    pEnglish = 0

    if (fSpanish - fEnglish) >= 100:
        pEnglish = 0
    elif (fSpanish - fEnglish) <= -100:
        pEnglish = 1
    else :
        pEnglish = 1 / (1 + math.e ** (fSpanish - fEnglish))
    
    print("Q4")
    print("%.4f" %pEnglish)

main()