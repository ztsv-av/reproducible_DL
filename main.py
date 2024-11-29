from model.train import train_model
from model.evaluate import evaluate_model

def main():
    print("\nStarting training...")
    train_accuracy = train_model(save=True)
    print(f"Done!\nModel accuracy on the training set: {train_accuracy:.2f}%")

    print("\nStarting evaluation...")
    test_accuracy = evaluate_model()
    print(f"Done!\nModel accuracy on the test set: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
