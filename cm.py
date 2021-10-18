import numpy as np

def compute_confusion_matrix(actual, predicted): # function with arguments
    uniqueclass = np.unique(actual)              # to check number of unique classes in both input arrays
    uniqueclassIterator  = uniqueclass           # to iterate through unique class
    dict = {}                                    # empty dictionary to collect the output

    for k in uniqueclassIterator:                #to iterate through class ID
        confusionmatrix = np.zeros((len(uniqueclass), len(uniqueclass))) # to create a numpy array of length of unique dimensions
        for i in range(len(uniqueclass)):        # nested for loop to calculate the each element of uniqueclass
            for j in range(len(uniqueclass)):
                confusionmatrix[i, j] = np.sum(((actual == uniqueclass[i]) &
                                               (predicted == uniqueclass[j])))

        uniqueclass = np.roll(uniqueclass,-1)    # np.roll to iterate through the classID
        #dict[k]=confusionmatrix                  #output of each classID in dictionary
        dict[k] = calculate_elements_confusionmatrix(confusionmatrix, k)  # output classid, tp,tn,fp,fn
    return dict


def calculate_elements_confusionmatrix(confusion_mat, i=0):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i,i]
    FP = confusion_mat[:,i].sum() - TP
    FN = confusion_mat[i,:].sum() - TP
    TN = confusion_mat.sum().sum() - TP - FP - FN
    return TP, FP, FN, TN


actual = [1,0,1,0,3,1,1,2,1]
predicted = [1,1,0,0,3,0,0,2,2]
result = compute_confusion_matrix(actual,predicted)
print(result)

