from Classifier import *

# train_file_path = input("Train file path : ")
# k_value = int(input("K value : "))

train_file_path = '..\\data\\train.txt'
k_value = 3

classifier = Classifier(train_file_path, k_value)

print("Labels set : ",classifier.labels)
print("Observations : ", len(classifier.observations))

# case_for_classification = input("Case for classification : ")

# ..\data\train.txt
