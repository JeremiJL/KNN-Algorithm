from Observation import *


def calculate_distance(observation_a, observation_b):
    distance = 0
    # Because we only compare distances with each other, there is no need to perform calculations like square
    # root of distance, also we can replace power function for absolute function
    for val_a, val_b in zip(observation_a.values, observation_b.values):
        distance += abs(val_a - val_b)

    return distance


def remove_entry_by_value(dictionary, value_to_remove):
    for key, value in dictionary.items():
        if value == value_to_remove:
            del dictionary[key]
            break


def return_key_by_value(dictionary, searched_value):
    for key, value in dictionary.items():
        if value == searched_value:
            return key


class Classifier:

    def __init__(self, train_file_path, k_val):
        # Path of file containing train data
        self.train_file_path = train_file_path
        # Value of k
        self.k_val = k_val
        # Labels tuple
        self.labels = self.extract_labels()
        # Observations tuple
        self.observations = self.extract_observations(train_file_path)
        # Test observations tuple
        self.test_observations = ()
        # Number of attributes
        self.num_of_attributes = self.observations[0].num_of_attributes

    def extract_labels(self):
        # Store labels in set, to keep them distinct
        labels_set = set()

        # Open file with train data
        with open(self.train_file_path, "r", encoding="UTF-8") as file:
            for line in file:
                labels_set.add(line.split(",")[-1].strip("\n"))

        # Collects all distinct labels that match the syntax of train data file
        return tuple(labels_set)

    def extract_observations(self, file_path):
        # Store observations in a list
        observations_list = []

        # Open file with train data
        with open(file_path, "r", encoding="UTF-8") as file:
            for line in file:
                o_data = line.split(",")
                observation = Observation(o_data[-1].strip("\n"), [float(val) for val in o_data[0:-1]])
                observations_list.append(observation)

        # Collect all observations from train data file into tuple
        return tuple(observations_list)

    def evaluate_accuracy(self, test_file_path, share_each_result):
        # Extract data from test file into tuple
        self.test_observations = self.extract_observations(test_file_path)

        # Store number of all test samples and number of successful test samples
        num_of_all_samples = len(self.test_observations)
        num_of_successes = 0

        # Classify each observation from test file and compare result with actual label
        for sample in self.test_observations:
            result = self.classify(sample)
            if result == sample.label:
                num_of_successes += 1
            # Print result of each sample according to method parameter
            if share_each_result:
                print(str(sample) + "\n\tClassified as " + str(result) + ". Success - " + str(result == sample.label))

        # Calculate accuracy based on number of successful test samples
        accuracy = num_of_successes * 1. / num_of_all_samples

        return accuracy

    def classify(self, new_observation):

        # Dictionary maps labels and number of occurrences in the neighborhood
        neighborhood = {}
        for label in self.labels:
            neighborhood[label] = 0

        # Dictionary maps observations and the k-th smallest distances
        distances = {}

        # Calculate distance between new observation and each observation from a train set
        for observation in self.observations:
            distance = calculate_distance(observation, new_observation)
            # Add distance into dictionary if it's distance is smaller than the greatest one currently stored
            # Make sure that exactly k distances are stored
            if len(distances.keys()) <= self.k_val or distance < max(distances.values()):
                distances[observation] = distance
            if len(distances.keys()) > self.k_val:
                remove_entry_by_value(distances, max(distances.values()))

        # Update neighbourhood dictionary
        for observation in distances.keys():
            neighborhood[observation.label] += 1

        # Find most frequently occurring label
        classified_label = return_key_by_value(neighborhood, max(neighborhood.values()))

        return classified_label
