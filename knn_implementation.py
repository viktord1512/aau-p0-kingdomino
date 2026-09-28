import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. SETUP OG KONSTANTER
DATASET_PATH = "dataset"
GROUND_TRUTH_PATH = "ground_truth.json"

TRAIN_BOARDS = ['1', '3', '4', '5', '7', '9', '10', '11', '12', '14', '18', '19', '20', '21', '23', '24', '25', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '47', '49', '50', '52', '53', '54', '57', '58', '59', '60', '61', '63', '66', '67', '69', '71', '72', '74']

VALIDATION_BOARDS = ['2', '13', '15', '16', '43', '45', '46', '48', '51', '64', '73']

TEST_BOARDS = ['6', '8', '17', '22', '26', '27', '55', '56', '62', '65', '68', '70']

# 2. INDLÆS DATA & FORBERED DATA
def load_data(board_numbers, ground_truth):
    X = []
    y = []
    for board_num in board_numbers:
        image_path = os.path.join(DATASET_PATH, f"{board_num}.jpg")
        image = cv2.imread(image_path)
        
        if image is None:
            continue
            
        board_labels = ground_truth[str(board_num)]
        
        for row in range(5):
            for col in range(5):
                # Klip feltet ud fra spillepladen (100x100 pixels)
                tile = image[row*100:(row+1)*100, col*100:(col+1)*100]
                
                # Konverter til HSV
                hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
                
                # Find medianfarven for H, S og V
                h, s, v = np.median(hsv_tile, axis=(0, 1))
                
                X.append([h, s, v])
                y.append(board_labels[row][col])
                
    return np.array(X), np.array(y)

def main():
    print("Indlæser data og facitliste...")
    with open(GROUND_TRUTH_PATH, "r") as file:
        ground_truth = json.load(file)

    # Indlæs trænings- og testdata
    X_train, y_train = load_data(TRAIN_BOARDS, ground_truth)
    X_val, y_val = load_data(VALIDATION_BOARDS, ground_truth)
    X_test, y_test = load_data(TEST_BOARDS, ground_truth)

    # 3. MODEL TRÆNING

    # Standardiser features, så algoritmen vægter H, S og V ligeligt
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    # Find den bedste K værdi
    best_k = 0
    best_accuracy = 0

    # Loop igennem alle ulige K værdier fra 1 til 50
    for k in range(1, 51, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_std, y_train)

        predictions = knn.predict(X_val_std)
        accuracy = accuracy_score(y_val, predictions)

        print(f"K={k}: {accuracy:.2%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"Den bedste K værdi er {best_k} med en accuracy på {best_accuracy:.2%}")

    # 4. TRÆN DEN ENDELIGE MODEL

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_std, y_train)

    # 5. TEST DEN ENDELIGE MODEL

    test_predictions = knn.predict(X_test_std)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Test accuracy: {test_accuracy:.2%}")

    # 6. VISUALISER CONFUSION MATRIX
    print("\nGenererer Confusion Matrix...")
    
    # 'normalize=true' regner antallet af felter om til procenter per række
    cm = confusion_matrix(y_test, test_predictions, labels=knn.classes_, normalize='true')
    
    # Vi sætter størrelsen på figuren
    fig, ax = plt.subplots(figsize=(10, 8))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
    
    # Vis grafen i procenter (values_format='.1%') med blåt farvetema
    disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax, values_format='.1%')
    
    # Tilføj en overskrift med testens nøjagtighed
    plt.title(f"Confusion Matrix for Kingdomino\nSamlet nøjagtighed (K={best_k}): {test_accuracy:.2%}")
    plt.tight_layout()
    plt.show()

# Kør programmet
if __name__ == "__main__":
    main()
