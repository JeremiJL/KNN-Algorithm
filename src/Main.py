import numpy as np
from Classifier import *
import matplotlib.pyplot as plt
import matplotlib
from ClassifierPlus import *

trainDataFile = "../data/train.txt"
testDataFile =  "../data/test.txt"

ClassifierPlus(trainDataFile,2).plot_accuracy_as_a_function_of_k(testDataFile,12)

