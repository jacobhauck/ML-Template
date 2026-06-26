import mlx
import torch
import torchvision.transforms.v2.functional as tvt
from PIL import Image
import matplotlib.pyplot as plt


@mlx.experiment
def run_siren_experiment(config, name, group=None):
    model = mlx.create_module(config['model'])
    image = tvt.to_image(Image.open(config['image_file']).convert('RGB'))
    image = tvt.to_dtype(image, torch.float) / 255.0
    x = torch.linspace(-1, 1, image.shape[2])
    y = torch.linspace(-1, 1, image.shape[1])
    xy = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
    xy = xy.reshape(-1, 2)

    mse_loss = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(config['iterations']):
        optim.zero_grad()
        pred = model(xy).T.reshape(image.shape)
        loss = mse_loss(pred[None], image[None])
        loss.backward()
        optim.step()
        print(f'Iteration {i} loss = {loss.item():.05f}')

    model.train(False)

    with torch.no_grad():
        pred = model(xy).T.reshape(image.shape).detach().cpu()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[1].imshow(pred.permute(1, 2, 0))
    plt.show()

    x = torch.linspace(-1, 1, 2*image.shape[2])
    y = torch.linspace(-1, 1, 2*image.shape[1])
    xy = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
    xy = xy.reshape(-1, 2)
    with torch.no_grad():
        pred = model(xy).T.reshape(3, 2*image.shape[1], 2*image.shape[2]).detach().cpu()
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[1].imshow(pred.permute(1, 2, 0))
    axes[2].imshow(Image.open(config['image_file']).convert('RGB').resize((2 * image.shape[2], 2 * image.shape[1])))
    plt.show()

