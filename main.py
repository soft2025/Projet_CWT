import argparse
import os
import torch
from src.models.swin import get_swin_tiny_partial_finetune
from src.train import create_dataloaders, train_model
from src.eval import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="CWT Swin Training")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to image dataset root')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to store training artifacts',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    num_classes = len(train_loader.dataset.dataset.class_to_idx)
    print(train_loader.dataset.dataset.class_to_idx)
    model = get_swin_tiny_partial_finetune(num_classes=num_classes)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'model.pth')

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=model_path,
    )
    evaluate_model(model, test_loader, device, class_names=list(train_loader.dataset.dataset.class_to_idx.keys()))


if __name__ == '__main__':
    main()
