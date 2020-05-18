#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import math
import copy
import numpy as np
import ast
import sys
import matplotlib.pyplot as plt


# In[3]:


def zero_one_loss(predicted, label):
    if predicted == label:
        return 0
    else:
        return 1


# In[12]:


def squared_loss(yes_prob, no_prob, label):
    if label == True:
        return (1 - yes_prob)**2
    else:
        return (1 - no_prob)**2

def preprocess_csv(filename):
    df = pd.read_csv(filename, keep_default_na=False)
    
    column_names = ['ambience', 'parking', 'dietaryRestrictions', 'recommendedFor']

    headers = []
    for i in list(df):
        if i not in column_names:
            headers.append(i)

    for name in column_names:
        l = []
        for i in df[name]:
            if i not in (None, ""):
                i = ast.literal_eval(i) 
                l.extend(i)
        l = list(set(l))
        headers.extend(l)
        
    data = {}

    for head in headers:
        data[head] = []

    # Iterates on all records in csv
    for index, row in df.iterrows():
        # Checks which header are added to new dataframe

        checklist_header = []
        for head in list(df):

            # Checks for non-binary columns
            if head not in column_names:

                # Appends entry if not None
                if row[head] not in (None, "", 'none'):
                    data[head].append(row[head])

                # Appends None if entry is empty
                else:
                    data[head].append('None')
                checklist_header.append(head)

            # Manipulates data for binary columns
            else:
                if row[head] not in (None, ""):
                    binary_data = ast.literal_eval(row[head])
                else:
                    binary_data = []

                for entry in binary_data:
                    data[entry].append(True)
                    checklist_header.append(entry)

        for head in headers:
            if head not in checklist_header:
                data[head].append(False)
                
    df = pd.DataFrame(data = data, columns = headers)
    return df


# In[3]:


# df = pd.read_csv("./output_data.csv", keep_default_na=False)
yelp_data = preprocess_csv("./yelp_data.csv")
fraction = []
one_zero_error = []
square_error = []
start = 0.05
for i in range(1, 20):
    fraction.append(start * i)

for f in fraction:
    print("")
    print("For Train Data Fraction = ", f)

    msk = np.random.rand(len(yelp_data)) < f
    train = yelp_data[msk]
    test = yelp_data[~msk]
    df = train
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]

    # df = preprocess_csv("./train_data.csv")
    # test = preprocess_csv("./test_data.csv")

    # df = preprocess_csv(train_file)
    # test = preprocess_csv(test_file)

    total_row = df.shape[0]
    test_row = test.shape[0]

    # print(len(test))
    # print(len(df))


    # In[4]:


    headers = list(df)
    to_predict = 'outdoorSeating'


    # In[5]:


    # Extract different types of values for each Attribute
    header_types = {}
    for head in headers:
        header_types[head] = {
            'types' : [],
            'num_types' : 0,
        }

    for index, row in df.iterrows():
        for head in headers:
            if row[head] not in header_types[head]['types'] and row[head] not in ['None', to_predict]:
                header_types[head]['types'].append(row[head])
                header_types[head]['num_types'] += 1


    # In[6]:


    posterior_count = {
        'yes' : 0,
        'no' : 0,
    }

    for index, row in df.iterrows():
        if row['outdoorSeating'] == True:
            posterior_count['yes'] += 1
        else:
            posterior_count['no'] += 1


    # In[7]:


    '''
        Structure of Probability Dictionary :-
        probability = {
            header_1 : {
                yes : {
                    attr1 : 0,
                    attr2 : 0,
                    .
                    .
                },
                no : {
                    attr1 : 0,
                    attr2 : 0,
                    .
                    .
                }
            }
            .
            .
        }
    '''


    # In[8]:


    probab_count = {}
    counts = {}

    for key, value in header_types.items():
        probab_count[key] = {
            'yes' : {},
            'no' : {},
        }

        counts[key] = {
            'yes' : 0,
            'no' : 0,
        }

        for val in value['types']:
            probab_count[key]['yes'][val] = 0
            probab_count[key]['no'][val] = 0

    for index, row in df.iterrows():
        for key, value in header_types.items():
            if row[key] != 'None':
                if row[to_predict] == True:
                    probab_count[key]['yes'][row[key]] += 1
                    counts[key]['yes'] += 1
                elif row[to_predict] == False:
                    probab_count[key]['no'][row[key]] += 1
                    counts[key]['no'] += 1


    # In[9]:


    # Calculate prior and posterior probabilities
    probability = copy.deepcopy(probab_count)

    for key, value in header_types.items():
        probability[key]['yes']['others'] = 1/ (counts[key]['yes'] + header_types[key]['num_types'])
        probability[key]['no']['others'] = 1/ (counts[key]['no'] + header_types[key]['num_types'])

        for key_attr, value in probability[key]['yes'].items():

            probability[key]['yes'][key_attr] = (probability[key]['yes'][key_attr] + 1)/(counts[key]['yes'] + header_types[key]['num_types']) 
            probability[key]['no'][key_attr] = (probability[key]['no'][key_attr] + 1)/(counts[key]['no'] + header_types[key]['num_types'])

    posterior_probab = copy.deepcopy(posterior_count)

    posterior_probab['yes'] /= total_row
    posterior_probab['no'] /= total_row


    # In[10]:


    # print(probability)
    # print(posterior_probab)


    # In[11]:





    # In[13]:


    correct = 0
    zero_loss = 0
    SE_loss = 0

    for index, row in test.iterrows():
        yes = posterior_probab['yes']
        no = posterior_probab['no']

        for key, value in header_types.items():
            if row[key] != 'None' and row[key] in probability[key]['yes']:
                yes *= probability[key]['yes'][row[key]]
                no *= probability[key]['no'][row[key]]
            else:
                yes *= probability[key]['yes']['others']
                no *= probability[key]['no']['others']

        total = yes + no
        yes /= total
        no /= total    

        if yes > no:
            ans = True
        else:
            ans = False

        if ans == row[to_predict]:
            correct += 1

        zero_loss += zero_one_loss(ans, row[to_predict])
        SE_loss += squared_loss(yes, no, row[to_predict])

    print("ZERO-ONE LOSS = ", zero_loss/test_row)
    print("SQUARED LOSS = ", SE_loss/test_row)

    one_zero_error.append(zero_loss/test_row)
    square_error.append(SE_loss/test_row)

    # In[14]:


    # df.to_csv('train_data.csv', index=False)
    # test.to_csv('test_data.csv', index=False)


# In[9]:


baseline_count = {'yes': 7179, 'no': 14820}
# Here since number of no is greater than yes, the baseline will be that outdoorSeating = False, this will give us a baseline zero-one error of yes/(yes+no) = 0.326, which is very high as compared to the bayes classifier
baseline_error = baseline_count['yes']/(baseline_count['yes']+baseline_count['no'])


# In[27]:


plt.plot(fraction, square_error, '-b', label='Square Error')
plt.plot(fraction, one_zero_error, '-r', label='Zero-One Error')
plt.xlabel('Fraction for Train Data', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.legend(loc='upper right')
plt.show()
# plt.plot(fraction, [baseline_error]*len(fraction))


# In[28]:


plt.plot(fraction, [baseline_error]*len(fraction), '-b', label='Baseline Error')
plt.plot(fraction, one_zero_error, '-r', label='Naive Bayes Error 0/1')
plt.xlabel('Fraction for Train Data', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.legend(loc='upper right')
plt.show()

