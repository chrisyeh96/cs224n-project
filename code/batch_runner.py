import csv
import subprocess

def createResultsCSV():
    with open("../results/model_results.csv", 'w') as f:
        fieldnames = ["cell", "distance_measure", "augment_data", "regularization_constant", "hidden_size", \
            "max_length", "best_dev_accuracy", "dev_f1", "test_accuracy", "test_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

if __name__ == "__main__":
    cells = ["rnn", "gru"]
    distance_measures = ["l2", "cosine", "custom_coef", "concat", "concat_steroids"]
    regularization_constants = [0.01, 0.001, 0.0001]
    hidden_sizes = [200, 250, 300]
    max_lengths = [20, 25, 30]

    for c in cells:
        for d in distance_measures:
            if (c == "gru") and (d in ["concat", "concat_steroids"]):
                augment_datas = [True, False]
            else:
                augment_datas = [False]

            for a in augment_datas:
                for r in regularization_constants:
                    for hs in hidden_sizes:
                        for ml in max_lengths:
                            command = "python run.py"

                            if augment_datas:
                                command += " -a"

                            command += " -c %s" % c
                            command += " -d %s" % d
                            command += " -r %g" % r
                            command += " -hs %d" % hs
                            command += " -ml %d" % ml

                            subprocess.call(command, shell=True)