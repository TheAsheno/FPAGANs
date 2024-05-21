import os
from random import random

image_dir="dataset/CACD2000/image/expriment"
list_dir="dataset/CACD2000/label"

def age_to_group(age):
    if age<=20:
        return 0
    elif age>20 and age<=30:
        return 1
    elif age>30 and age<=40:
        return 2
    elif age>40 and age<=50:
        return 3
    elif age>50:
        return 4

train_txt=[]
test_txt=[]
train_age_group0=[]
train_age_group1=[]
train_age_group2=[]
train_age_group3=[]
train_age_group4=[]
test_age_group0=[]
test_age_group1=[]
test_age_group2=[]
test_age_group3=[]
test_age_group4=[]



for filename in os.listdir(image_dir):
    age=int(filename.split('_')[0])
    group=age_to_group(age)
    strline = filename + ' %d\n' % group
    if random()<0.9:
        if group==0:
            train_age_group0.append(strline)
        elif group==1:
            train_age_group1.append(strline)
        elif group==2:
            train_age_group2.append(strline)
        elif group==3:
            train_age_group3.append(strline)
        elif group==4:
            train_age_group4.append(strline)
        train_txt.append(strline)
    else:
        if group==0:
            test_age_group0.append(strline)
        elif group==1:
            test_age_group1.append(strline)
        elif group==2:
            test_age_group2.append(strline)
        elif group==3:
            test_age_group3.append(strline)
        elif group==4:
            test_age_group4.append(strline)
        test_txt.append(strline)

with open(list_dir + '/train_age_group_0.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group0)

with open(list_dir + '/train_age_group_1.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group1)

with open(list_dir + '/train_age_group_2.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group2)

with open(list_dir + '/train_age_group_3.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group3)

with open(list_dir + '/train_age_group_4.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group4)

with open(list_dir + '/train.txt','w', encoding='utf-8') as f:
    f.writelines(train_txt)

with open(list_dir + '/test_age_group_0.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group0)

with open(list_dir + '/test_age_group_1.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group1)

with open(list_dir + '/test_age_group_2.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group2)

with open(list_dir + '/test_age_group_3.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group3)

with open(list_dir + '/test_age_group_4.txt','w', encoding='utf-8') as f:
    f.writelines(train_age_group4)

with open(list_dir + '/test.txt','w', encoding='utf-8') as f:
    f.writelines(test_txt)