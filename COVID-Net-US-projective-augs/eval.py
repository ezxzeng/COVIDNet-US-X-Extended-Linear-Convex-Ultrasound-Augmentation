import glob
import json
import os
import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

from dataset import get_test_dataloader
from helper import get_hyperparameter, DEVICE

from darwin_net.generate_model import main as pretrained_darwin_model


results_json_path = "runs/results.json"


def update_results_json(all_results_json, results_dict, model_path):
    results_keys = os.path.dirname(model_path).split("/")
    intermediate_json = all_results_json
    for key in results_keys:
        if key not in intermediate_json:
            intermediate_json[key] = {}
        intermediate_json = intermediate_json[key]

    intermediate_json["results"] = results_dict

    return all_results_json


def evaluate_model(model_path, nb_classes, dataloader, criterion, dataset_size, all_results_json):
    model = torch.load(model_path)
    model.eval()

    running_loss = 0.0
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    print("evaluating: ", model_path)

    all_labels = torch.zeros(0)
    all_probs = torch.zeros(0)

    # Iterate over data.
    for inputs, labels in tqdm.tqdm(dataloader):
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)[:, 1]

        # statistics
        running_loss += loss.item() * inputs.size(0)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        probs = probs.to("cpu")
        labels = labels.to("cpu")

        all_labels = torch.cat((all_labels, labels))
        all_probs = torch.cat((all_probs, probs))

    auc = roc_auc_score(all_labels, all_probs)

    loss = running_loss / dataset_size
    acc = confusion_matrix.diag().sum() / confusion_matrix.sum()
    print(acc)

    tp = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    recall = tp.double() / (fn.double() + tp.double())
    precision = tp.double() / (fp.double() + tp.double())
    f1 = 2 * precision * recall / (precision + recall)

    save_path = os.path.join(
        os.path.dirname(model_path), "results_acc{:.2f}_auc{:.2f}.json".format(acc * 100, auc*100)
    )

    results_dict = {
                "accuracy": float(acc),
                "auc": float(auc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": confusion_matrix.tolist(),
                "model_path": model_path
            }

    with open(save_path, "x") as f:
        json.dump(results_dict, f, indent=4)

    all_results_json = update_results_json(all_results_json, results_dict, model_path)

    with open(results_json_path, "w") as f:
        json.dump(all_results_json, f, indent=4)

    return all_results_json


if __name__ == "__main__":
    hp = get_hyperparameter(hp_path="eval_hyperparameters.json")

    model_paths = []
    for path in hp["model_paths"]:
        model_paths.extend(glob.glob(path))

    with open(results_json_path, "r") as f:
        all_results_json = json.load(f)

    dataloader, dataset_size = get_test_dataloader(**hp["data_params"])
    nb_classes = hp["model_params"]["num_classes"]

    # Loss function
    criterion = nn.CrossEntropyLoss()

    for model_path in model_paths:
        all_results_json = evaluate_model(model_path, nb_classes, dataloader, criterion, dataset_size, all_results_json)
        
