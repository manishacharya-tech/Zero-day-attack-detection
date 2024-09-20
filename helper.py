import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import random


def balancing(df):
    benign_total = len(df[df['Label'] == "BENIGN"])
    attack_total = len(df[df['Label'] != "BENIGN"])

    if attack_total == 0:  # If there are no attack records
        return df  # Return the original DataFrame

    enlargement = 1.1
    benign_included_max = attack_total / 30 * 70
    benign_inc_probability = (benign_included_max / benign_total) * enlargement

    indexes = []
    benign_included_count = 0

    for index, row in df.iterrows():
        if row['Label'] != "BENIGN":
            indexes.append(index)
        else:
            if random.random() > benign_inc_probability:
                continue
            if benign_included_count > benign_included_max:
                continue
            benign_included_count += 1
            indexes.append(index)

    df_balanced = df.loc[indexes]
    return df_balanced


def merge_attack_classes(df):
    # Define new labels based on the old labels
    label_mapping = {
        'Benign': 'Normal',
        'Bot': 'Botnet',
        'FTP-Patator': 'BruteForce',
        'SSH-Patator': 'BruteForce',
        'DDoS': 'Dos/DDos',
        'DoS GoldenEye': 'Dos/DDos',
        'DoS Hulk': 'Dos/DDos',
        'DoS Slowhttptest': 'Dos/DDos',
        'DoS slowloris': 'Dos/DDos',
        'Heartbleed': 'Dos/DDos',
        'Infiltration': 'Infiltration',
        'PortScan': 'PortScan',
        'Web Attack – Brute Force': 'WebAttack',
        'Web Attack – Sql Injection': 'WebAttack',
        'Web Attack – XSS': 'WebAttack'
    }

    # Apply the mapping
    df['Label'] = df['Label'].replace(label_mapping)
    return df


def replace_default_and_infinite_values(df, threshold=0.01, replacement_value=0):
    for column in df.columns:
        nan_ratio = df[column].isna().mean()
        inf_ratio = (df[column] == np.inf).mean() + (df[column] == -np.inf).mean()

        if nan_ratio <= threshold:
            df[column] = df[column].fillna(replacement_value)

        if inf_ratio <= threshold:
            df[column] = df[column].replace([np.inf, -np.inf], replacement_value)

    return df


def dataframe_drop_correlated_columns(df, threshold=0.90, verbose=False):
    if verbose:
        print('Dropping correlated columns')
    if threshold == -1:
        return df, []

    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    df = df.drop(df[to_drop], axis=1)

    return df, to_drop


def file_write_args(args, file_name, one_line=False):
    args = vars(args)

    with open(file_name, "a") as file:
        file.write('BEGIN ARGUMENTS\n')
        if one_line:
            file.write(str(args))
        else:
            for key in args.keys():
                file.write('{}, {}\n'.format(key, args[key]))

        file.write('END ARGUMENTS\n')


def plot_model_history(hist, output_file):
    plt.clf()
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'], loc='center right')
    plt.savefig(output_file)


def plot_accuracy_per_threshold(thr, attack_path, output_file):
    plt.figure(figsize=(10, 6))
    plt.bar(thr.keys(), thr.values(), width=0.01, align='center', alpha=0.7)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Threshold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'_accuracy_per{attack_path}.png')
    # plt.savefig(f'{output_file.replace(".csv", "_accuracy_per_threshold.png")}')
    plt.close()

def plot_accuracy_vs_threshold(threshold,accuracies, attack_path):
    plt.figure()
    plt.plot(threshold, accuracies, marker='o', color='blue', lw=2)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Threshold - {attack_path}')
    plt.grid(True)
    plt.savefig(attack_path.replace('.csv', '_accuracy_vs_threshold.pdf'))
    plt.show()


def plot_roc(fpr, tpr, auc, attack_path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {attack_path}')
    plt.legend(loc="lower right")
    plt.savefig(attack_path.replace('.csv', '_roc_curve.pdf'))
    plt.show()
    print(f'AUC for {attack_path}:', auc)


def plot_tnr_th(threshold, true_negative_rate,attack_path):
    plt.figure()
    plt.plot(threshold, true_negative_rate, marker='o', color='green', lw=2)
    plt.xlabel('Threshold')
    plt.ylabel('True Negative Rate (TNR)')
    plt.title(f'True Negative Rate vs Threshold - {attack_path}')
    plt.grid(True)
    plt.savefig(attack_path.replace('.csv', '_tnr_vs_threshold.pdf'))
    plt.show()
