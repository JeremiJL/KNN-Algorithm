from Observation import *

class Classifier:

    def __init__(self, train_file_path, k_val):
        # Path of file containing train data
        self.train_file_path = train_file_path
        # Value of k
        self.k_val = k_val
        # Labels tuple
        self.labels = self.extract_labels()
        # Observations tuple
        self.observations = self.extract_observations()

    def extract_labels(self):
        # Store labels in set, to keep them distinct
        labels_set = set()

        # Open file with train data
        with open(self.train_file_path, "r", encoding="UTF-8") as file:
            for line in file:
                labels_set.add(line.split(",")[-1].strip("\n"))

        # Collects all distinct labels that match the syntax of train data file
        return tuple(labels_set)

    def extract_observations(self):
        # Store observations in a list
        observations_list = []

        # Open file with train data
        with open(self.train_file_path, "r", encoding="UTF-8") as file:
            for line in file:
                o_data = line.split(",")
                observation = Observation(o_data[-1].strip("\n"), o_data[0:-1])
                observations_list.append(observation)

        # Collect all observations from train data file into tuple
        return tuple(observations_list)
