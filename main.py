import argparse
import pandas as pd
import helper
from sklearn import preprocessing
import os
import numpy as np
import autoencoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def evaluate(model, valid_X, y_true, attack_path, output_file):
    threshold = np.arange(0.05, 2, 0.05)
    y_pred_score = model.predict(valid_X)
    val_reconstruction_error = (np.mean(np.square(valid_X - y_pred_score), axis=1))
    accuracies = []
    true_negative_rate = []
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, val_reconstruction_error)
    auc = roc_auc_score(y_true, val_reconstruction_error)
    for th in threshold:
        with open(output_file, "a") as file:
            file.write(f'Results for {attack_path}\n')
            file.write('Threshold, Precision, Recall, F1-Score, Support\n')
            y_pred = (val_reconstruction_error > th).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            accuracies.append(accuracy)
            unique_classes = np.unique(y_true)
            num_classes = len(unique_classes)
            target_names = ['Benign', 'Anomaly'][:num_classes]
            labels = unique_classes
            report = classification_report(y_true, y_pred, target_names=target_names,labels=labels, output_dict=True, zero_division=True)
            print(f'for {attack_path}:', report)
            cm = confusion_matrix(y_true, y_pred)
            print(f'for {attack_path}:', cm)
            if num_classes == 2:  # Check if both classes are present
                precision = report['Anomaly']['precision']
                recall = report['Anomaly']['recall']
                f1_score = report['Anomaly']['f1-score']
                support = report['Anomaly']['support']
            else:
                precision = recall = f1_score = support = 0.0
            file.write(f'{th}, {precision:.4f}, {recall:.4f}, {f1_score:.4f}, {support}\n')
            tn = cm[0][0]  # True Negatives
            fp = cm[0][1]  # False Positives
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            true_negative_rate.append(tnr)

    helper. plot_accuracy_vs_threshold(threshold, accuracies, attack_path)
    # Plot True Negative Rate vs Threshold
    helper.plot_tnr_th(threshold, true_negative_rate, attack_path)
    # Plot ROC Curve
    helper.plot_roc(fpr,tpr,auc, attack_path)

def load_and_prepr  ocess_data(normal_path, correlation_value):
    df = pd.read_csv(normal_path, encoding='cp1252')
    df.columns = df.columns.str.replace(' ', '')
    df = helper.merge_attack_classes(df)
    attack_or_not = [0 if i == "BENIGN" else 1 for i in df["Label"]]
    df["Label"] = attack_or_not
    df = helper.replace_default_and_infinite_values(df)
    y = df["Label"].astype(np.int64)
    df.drop(['Label'], axis=1, inplace=True)
    normal, to_drop = helper.dataframe_drop_correlated_columns(df, correlation_value)
    pipeline = Pipeline([('normalizer', preprocessing.Normalizer()), ('scaler', preprocessing.StandardScaler())])
    dense_matrix_scaled = pd.DataFrame(pipeline.fit_transform(normal.values))
    return dense_matrix_scaled, y, to_drop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_path', default='MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv')
    parser.add_argument('--attack_paths', default='MachineLearningCVE')
    parser.add_argument('--output', default='Results.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--regu', default='l2')
    parser.add_argument('--l1_value', type=float, default=0.01)
    parser.add_argument('--l2_value', type=float, default=0.0001)
    parser.add_argument('--correlation_value', type=float, default=0.9)
    parser.add_argument('--model', default='autoencoder')
    parser.add_argument('--loss', default='mse')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=54460)
    parser.add_argument("--host", default='127.0.0.1')
    args = parser.parse_args()
    output_file = args.output
    output_file = datetime.now().strftime("%d_%m_%Y__%H_%M_") + output_file
    helper.file_write_args(args, output_file)
    normal, y, to_drop = load_and_preprocess_data(args.normal_path, args.correlation_value)
    X_train, X_val, y_train,y_val = train_test_split(normal,y,test_size=.2,train_size=.8,shuffle=True,random_state=42)
    wrapper = autoencoder.Autoencoder(num_features=normal.shape[1], learning_rate=1e-5)
    with open(output_file, 'a', encoding="utf-8") as file:
        wrapper.model.summary(print_fn=lambda x: file.write(x + '\n'))

    hist = wrapper.train(X_train, X_val)
    helper.plot_model_history(hist, f'{output_file.replace(".csv", f"training.pdf")}')

    res1 = wrapper.model.evaluate(X_train, y_train)
    res2 = wrapper.model.evaluate(X_val, y_val)
    with open(output_file, "a", ) as file:
        file.write(f'Training accuracy, {res1[1]} \n Validation accuracy, {res2[1]}\n')

    evaluate(wrapper.model, X_val, y_val, f'Validation', output_file)


    for attack_path in os.listdir(args.attack_paths):
        path = os.path.join(args.attack_paths, attack_path)
        attack = pd.read_csv(path)
        attack = attack.dropna()
        attack.columns = attack.columns.str.replace(' ', '')
        attack.drop(to_drop, axis=1, inplace=True)
        attack = helper.merge_attack_classes(attack)
        attack_or_not = [0 if i == "BENIGN" else 1 for i in attack["Label"]]
        attack["Label"] = attack_or_not
        attack = helper.replace_default_and_infinite_values(attack)
        y_attack = attack["Label"].astype(np.int64)
        attack.drop(['Label'], axis=1, inplace=True)
        pipeline = Pipeline([('normalizer', preprocessing.Normalizer()),
                             ('scaler', preprocessing.StandardScaler())])
        x_scaled = pd.DataFrame(pipeline.fit_transform(attack.values))
        evaluate(wrapper.model, x_scaled, y_attack, attack_path, output_file)
