from Classifier import *
import matplotlib.pyplot as plt

def draw_bar_plot(data_map):
    # Creating the data set
    y_values = list(data_map.values())
    x_values = list(data_map.keys())

    # Creating figure
    plt.figure(figsize=(10, 5))

    # Setting up bar plot
    plt.xticks(x_values)
    plt.grid(zorder=3)
    colors = ['teal' if value == 1 else 'gray' for value in y_values]
    plt.bar(x_values, y_values, color=colors, width=0.5)
    plt.xlabel("Values of K Parameter")
    plt.ylabel("Accuracy of Model")
    plt.title("Evaluated accuracy of model based on train data")

    # Save the plot as a PNG file
    plt.savefig('../plots/plot.png')

    plt.show()


class ClassifierPlus(Classifier):

    def __init__(self, train_file_path, k_val):
        super(ClassifierPlus, self).__init__(train_file_path, k_val)

    def plot_accuracy_as_a_function_of_k(self, test_file_path, max_k):
        # Map k values with their accuracy results from evaluation
        accuracy_against_k = {}

        # Calculate accuracies from k = 1 up to k = max_k and store them into dictionary
        for k_val in range(1, max_k):
            result = Classifier(self.train_file_path, k_val).evaluate_accuracy(test_file_path, False)
            accuracy_against_k[k_val] = result

        draw_bar_plot(accuracy_against_k)


