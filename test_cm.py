import cm               #importing cm file for unittesting

# Binary Test Case
def test_compute_confusion_matrix_binary():
    # From cm file call the defined function compute_confusion_matrix to compute confusionmatrix of the given input
    # which later calls calculate_elements_confusionmatrix function to calculate tp,tn,fp,fn.
    res = cm.compute_confusion_matrix([1,0,1,0,0,1,1], [1,1,0,0,1,0,0])
    assert res == {0: (1.0, 3.0, 2.0, 1.0), 1: (1.0, 3.0, 2.0, 1.0)}

# Multiclass Test Case
def test_compute_confusion_matrix_multiclass():
    # From cm file call the defined function compute_confusion_matrix to compute confusionmatrix of the given input
    # which later calls calculate_elements_confusionmatrix function to calculate tp,tn,fp,fn.
    res = cm.compute_confusion_matrix([1,0,1,0,3,1,1,2,1], [1,1,0,0,3,0,0,2,2])
    assert res == {0: (1.0, 3.0, 1.0, 4.0),
                   1: (1.0, 1.0, 0.0, 7.0),
                   2: (1.0, 3.0, 1.0, 4.0),
                   3: (1.0, 1.0, 0.0, 7.0)}
