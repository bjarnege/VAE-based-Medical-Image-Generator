import argparse
import pathlib

from tqdm.auto import trange, tqdm
import torch
import torch.nn
import medmnist.dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae_based_medical_image_generator.data import dataset
from vae_based_medical_image_generator.model.loss import vae_loss
from vae_based_medical_image_generator.model.vae import VariationalAutoencoder
from vae_based_medical_image_generator.model.cvae import ConditionalVariationalAutoencoder
from vae_based_medical_image_generator.metrics.kl_divergence import calc_kl_divergence
from vae_based_medical_image_generator.metrics.reconstruction import calc_reconstruction_loss_bce, calc_reconstruction_loss_mse


def run_model(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor):
    """
    Runs the given model on the input images and labels.

    Args:
        model (torch.nn.Module): The model to be run.
        images (torch.Tensor): Input images to be processed by the model.
        labels (torch.Tensor): Labels for the input images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the output image, the mean and the log variance.
    """
    if isinstance(model, VariationalAutoencoder):
        return model(images)
    else:
        return model(images, labels)


def reconstruct_image(model: torch.nn.Module, dataset: medmnist.dataset.MedMNIST, device: torch.device, sample_size: int = 1):
    """
    Reconstructs images using the given model and dataset.

    Args:
        model (torch.nn.Module): The model to be used for reconstruction.
        dataset (medmnist.dataset.MedMNIST): The dataset to use for reconstruction.
        device (torch.device): The device on which the model is run.
        sample_size (int, optional): The number of samples to reconstruct. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the original and reconstructed images.
    """

    data_loader = DataLoader(dataset, batch_size=sample_size, shuffle=True)

    with torch.no_grad():
        original_images, labels = next(iter(data_loader))

        original_images = original_images.to(device)
        labels = labels.to(device)
        reconstructed_images, _, _ = run_model(model, original_images, labels)

        return original_images, reconstructed_images


def train(model: torch.nn.Module, train_dataloader: DataLoader[medmnist.dataset.MedMNIST], optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler, device: torch.device):
    """
    Trains the given model using the provided dataloader, optimizer and scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (DataLoader[medmnist.dataset.MedMNIST]): The dataloader for training data.
        optimizer (torch.optim.Optimizer): The optimizer to be used during training.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to be used during training.
        device (torch.device): The device on which the model is run.

    Returns:
        float: The average training loss per sample.
    """
    model.train()

    total_loss = 0
    pbar = tqdm(train_dataloader, unit="batch", desc="Training")
    for images, labels in pbar:
        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()

        output_images, mean, log_variance = run_model(model, images, labels)
        training_loss = vae_loss(images, output_images, mean, log_variance)
        total_loss += training_loss.item()

        training_loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_dataloader.dataset)


def evaluate(model: torch.nn.Module, validation_dataloader: DataLoader[medmnist.dataset.MedMNIST], device: torch.device):
    """
    Evaluates the given model using the provided dataloader.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        validation_dataloader (DataLoader[medmnist.dataset.MedMNIST]): The dataloader for validation data.
        device (torch.device): The device on which the model is run.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the average loss, KL divergence, reconstruction loss (binary cross-entropy), and reconstruction loss (mean squared error) per sample.
    """
    model.eval()

    loss = 0
    kl_divergence = 0
    reconstruction_loss_bce = 0
    reconstruction_loss_mse = 0

    with torch.no_grad():
        pbar = tqdm(validation_dataloader, unit="batch", desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            output_images, mean, log_variance = run_model(model, images, labels)

            loss += vae_loss(images, output_images, mean, log_variance).item()
            kl_divergence += calc_kl_divergence(mean, log_variance).item()
            reconstruction_loss_bce += calc_reconstruction_loss_bce(images, output_images).item()
            reconstruction_loss_mse += calc_reconstruction_loss_mse(images, output_images).item()

    loss /= len(validation_dataloader.dataset)
    kl_divergence /= len(validation_dataloader.dataset)
    reconstruction_loss_bce /= len(validation_dataloader.dataset)
    reconstruction_loss_mse /= len(validation_dataloader.dataset)
    return loss, kl_divergence, reconstruction_loss_bce, reconstruction_loss_mse


def fit(model: torch.nn.Module,
        model_dir: pathlib.Path,
        train_dataloader: DataLoader[medmnist.dataset.MedMNIST],
        validation_dataloader: DataLoader[medmnist.dataset.MedMNIST],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        num_epochs: int,
        checkpoint_every: int = 10,
        predict_every: int = 10):
    """
    Trains the given model using the provided dataloaders, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        model_dir (pathlib.Path): The directory to save the trained model.
        train_dataloader (DataLoader[medmnist.dataset.MedMNIST]): The dataloader for training data.
        validation_dataloader (DataLoader[medmnist.dataset.MedMNIST]): The dataloader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer to be used during training.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler to be used during training.
        device (torch.device): The device on which the model is run.
        num_epochs (int): The number of epochs to train the model.
        checkpoint_every (int, optional): The frequency of saving model checkpoints. Defaults to 10.
        predict_every (int, optional): The frequency of reconstructing and saving sample images. Defaults to 10.
    """

    writer = SummaryWriter(log_dir=model_dir / "logs")

    pbar = trange(1, num_epochs + 1, unit="epochs")
    for epoch in range(1, num_epochs + 1):
        pbar.set_description(f"Epoch {epoch} of {num_epochs}, LR: {scheduler.get_last_lr()}")

        training_loss = train(model, train_dataloader, optimizer, scheduler, device)
        writer.add_scalar('Loss/Train', training_loss, epoch)

        validation_loss, kl_divergence, reconstruction_loss_bce, reconstruction_loss_mse = evaluate(model, validation_dataloader, device)
        writer.add_scalar('Loss/Validation', validation_loss, epoch)
        writer.add_scalar('Metric/kl_divergence', kl_divergence, epoch)
        writer.add_scalar('Metric/reconstruction_bce', reconstruction_loss_bce, epoch)
        writer.add_scalar('Metric/reconstruction_mse', reconstruction_loss_mse, epoch)

        if epoch % predict_every == 0:
            original_image, reconstructed_image = reconstruct_image(model, train_dataloader.dataset, device)
            writer.add_images('Image/Training/Original and Reconstructed Images',
                              torch.cat((original_image, reconstructed_image), dim=-1),
                              epoch,
                              dataformats="NCHW")

            original_image, reconstructed_image = reconstruct_image(model, validation_dataloader.dataset, device)
            writer.add_images('Image/Validation/Original and Reconstructed Images',
                              torch.cat((original_image, reconstructed_image), dim=-1),
                              epoch,
                              dataformats="NCHW")

        if epoch % checkpoint_every == 0:
            torch.save(model.state_dict(), model_dir / f"checkpoint_{epoch}.pt")


def main(args):
    """
    The main function to train a VAE or CVAE model on the MedMNIST dataset.

    Args:
        args (argparse.Namespace): The command line arguments for the main function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset
    train_dataset = dataset.load_dataset(dataset_name=args.dataset, split='train')
    validation_dataset = dataset.load_dataset(dataset_name=args.dataset, split='val')

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    if args.model == 'vae':
        model = VariationalAutoencoder(image_channels=train_dataset.info["n_channels"],
                                       latent_dimension=args.latent_dimension,
                                       device=device).to(device)
    else:
        if not train_dataset[0][1].shape[0] > 1:  # check if data is already one hot encoded
            labels_train = torch.as_tensor(train_dataset.labels.squeeze()).to(torch.int64)
            train_dataset.labels = torch.nn.functional.one_hot(labels_train).numpy()

            labels_test = torch.as_tensor(validation_dataset.labels.squeeze()).to(torch.int64)
            validation_dataset.labels = torch.nn.functional.one_hot(labels_test).numpy()

        model = ConditionalVariationalAutoencoder(image_channels=train_dataset.info["n_channels"],
                                                  n_labels=len(train_dataset.info["label"]),
                                                  latent_dimension=args.latent_dimension,
                                                  device=device).to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * 50, gamma=0.5, last_epoch=-1, verbose=False)

    root_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
    model_dir = root_dir / "models" / args.dataset / args.model / f"lr-{args.lr}-bs-{args.batch_size}-latent-{args.latent_dimension}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    fit(model, model_dir, train_loader, validation_loader, optimizer, scheduler, device, num_epochs=args.num_epochs)


if __name__ == '__main__':
    # For debugging purposes
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    print(os.getpid())

    parser = argparse.ArgumentParser(description='Train a VAE or CVAE on the MedMNIST dataset.')

    parser.add_argument('--model',
                        type=str,
                        choices=['vae', 'cvae'],
                        default='vae',
                        help='Choose either VAE or CVAE to train (default: VAE)')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples per batch (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--latent_dimension', type=int, default=64, help='Size of the latent vector / bottleneck (default: 64)')
    parser.add_argument('--dataset',
                        type=str,
                        choices=[
                            'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist',
                            'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist'
                        ],
                        default='chestmnist',
                        help='Choose a dataset to train on (default: chestmnist)')

    main(parser.parse_args())