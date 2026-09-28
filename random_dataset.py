import json
import random

with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)

boards = list(ground_truth.keys())

random.seed(26)
random.shuffle(boards)

train_end = int(len(boards) * 0.7)
validation_end = int(len(boards) * 0.85)

train_boards = boards[:train_end]
validation_boards = boards[train_end:validation_end]
test_boards = boards[validation_end:]

print("Training:", sorted(train_boards, key=int))
print("Validation:", sorted(validation_boards, key=int))
print("Test:", sorted(test_boards, key=int))
