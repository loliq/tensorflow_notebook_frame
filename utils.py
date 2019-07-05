#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import re


# 1. 绘制混淆矩阵

# In[2]:


def plot_confusion_matrix(labels, y_true, y_pred,fontsize=20, title = "Confusion Matrix"):
    tick_marks = np.array(range(len(y_true) + len(y_pred))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        cp = cm_normalized[y_val][x_val]
        if (c > 0.01):
            plt.text(x_val, y_val, "%d" %(c,), color='red', fontsize=fontsize, va='top', ha='center')
            plt.text(x_val, y_val, "%0.4f %s" % (cp,'%'), color='red', fontsize=fontsize, va='bottom', ha='center')

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # show confusion matrix
    plt.show()


# 2. 将'\' 转为 ‘/’

# In[3]:


def trans_dir(oriDir):
    transformed_dir = re.sub(r'\\', '/', oriDir)
    return transformed_dir


# In[ ]:




