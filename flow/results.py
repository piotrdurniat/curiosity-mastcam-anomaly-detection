import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

def give_results(true_image_loss, fake_image_loss, save_path):
    # Calculate average loss
    avg_true_loss = np.mean(list(true_image_loss.values()))
    avg_fake_loss = np.mean(list(fake_image_loss.values()))

    print(f"Średnia strata dla prawdziwych obrazów: {avg_true_loss}")
    print(f"Średnia strata dla fałszywych obrazów: {avg_fake_loss}")

    # Determine which were considered true and which fake
    all_losses = {**true_image_loss, **fake_image_loss}
    true_labels = [1 if key in true_image_loss else 0 for key in all_losses]
    predicted_labels = [1 if loss < avg_true_loss else 0 for loss in all_losses.values()]

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fałszywe", "Prawdziwe"])

    # Plot confusion matrix
    cm_display.plot()
    plt.title('Macierz konfuzji')
    plt.savefig(f"{save_path}/conf_matrix.png")

    # Calculate F1 score and precision
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    with open(f"{save_path}/metrics.txt", "w") as file:
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 score: {f1}\n")