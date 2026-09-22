from Checkpointing import checkpoint_path, load_checkpoint, save_checkpoint
from Creating_Train_and_Test_set import create_dataset_for_5folds
from DTADataset import *
from Emetrics import *
from GNNNet import GNNNet
from Paths import paths_for

dataset = "davis"
folds = [0, 1, 2, 3, 4]

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
NUM_EPOCHS = 2000
# Checkpoint cadence in epochs.
CHECKPOINT_EVERY_EPOCHS = 1

print("Dataset: ", dataset)
print("Learning rate: ", LR)
print("Epochs: ", NUM_EPOCHS)

PATHS = paths_for(dataset)
models_dir = PATHS.models
results_dir = PATHS.results
print("Data root: ", PATHS.root)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Force using CPU
print("Using device:", device)

for fold in folds:
    model = GNNNet()
    model.to(device)
    model_st = GNNNet.__name__
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    checkpoint_file_name = checkpoint_path(models_dir, model_st, dataset, fold)
    start_epoch, best_mse, best_epoch = load_checkpoint(
        checkpoint_file_name, fold, model, optimizer, device
    )
    if start_epoch > 0:
        print(
            f"Resuming fold {fold} at epoch {start_epoch + 1} (best MSE {best_mse} at epoch {best_epoch})"
        )

    print(f"Training for fold {fold}...")

    train_data, valid_data = create_dataset_for_5folds(
        dataset_name=dataset, fold_idx=fold
    )
    train_loader = DataLoader(
        train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    valid_loader = DataLoader(
        valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    model_file_name = os.path.join(
        models_dir, f"model_{model_st}_{dataset}_{fold}.model"
    )

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print("Predicting for validation data...")
        G, P = predicting(model, device, valid_loader)
        val_mse = get_mse(G, P)
        print(f"Epoch {epoch + 1} - Validation MSE: {val_mse}, Best MSE: {best_mse}")
        improved = val_mse < best_mse
        if improved:
            best_mse = val_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print(f"MSE improved at epoch {best_epoch}; Best MSE: {best_mse}")
        else:
            print(f"No improvement since epoch {best_epoch}; Best MSE: {best_mse}")

        if improved or (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            save_checkpoint(
                checkpoint_file_name,
                fold,
                epoch + 1,
                model,
                optimizer,
                best_mse,
                best_epoch,
            )
