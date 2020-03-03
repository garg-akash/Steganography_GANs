import datetime
import matplotlib.pyplot as plt
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from critic import BasicCritic
from decoder import BasicDecoder
from encoder import BasicEncoder
from torchvision import datasets, transforms
from IPython.display import clear_output
import torchvision
from torch.optim import Adam
import pytorch_ssim
from tqdm import tqdm
import torch
import os
import gc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def plot(name, train_epoch, values, save):
    clear_output(wait=True)
    plt.close('all')
    fig = plt.figure()
    fig = plt.ion()
    fig = plt.subplot(1, 1, 1)
    fig = plt.title('epoch: %s -> %s: %s' % (train_epoch, name, values[-1]))
    fig = plt.ylabel(name)
    fig = plt.xlabel('epoch')
    fig = plt.plot(values)
    fig = plt.grid()
    get_fig = plt.gcf()
    fig = plt.draw()  # draw the plot
    fig = plt.pause(1)  # show it for 1 second
    if save:
        now = datetime.datetime.now()
        get_fig.savefig('results/plots/%s_%d_%d_%s.png' %
                        (name, train_epoch, values[-1], now.strftime("%Y-%m-%d_%H:%M:%S")))


def main():
    data_dir = 'div2k'
    epochs = 5
    data_depth = 2
    hidden_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc',
        'val.cover_score',
        'val.generated_score',
        'val.ssim',
        'val.psnr',
        'val.bpp',
        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc',
        'train.cover_score',
        'train.generated_score',
    ]

    mu = [.5, .5, .5]
    sigma = [.5, .5, .5]

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(
                                        360, pad_if_needed=True),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mu, sigma)])

    train_set = datasets.ImageFolder(os.path.join(
        data_dir, "train/"), transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True)

    valid_set = datasets.ImageFolder(os.path.join(
        data_dir, "val/"), transform=transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=4, shuffle=False)

    encoder = BasicEncoder(data_depth, hidden_size)
    decoder = BasicDecoder(data_depth, hidden_size)
    critic = BasicCritic(hidden_size)
    cr_optimizer = Adam(critic.parameters(), lr=1e-4)
    # Why add encoder parameters too?
    en_de_optimizer = Adam(list(decoder.parameters()) +
                           list(encoder.parameters()), lr=1e-4)

    for ep in range(epochs):
        metrics = {field: list() for field in METRIC_FIELDS}
        for cover, _ in tqdm(train_loader):
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            cover_score = torch.mean(critic.forward(cover))
            generated_score = torch.mean(critic.forward(generated))

            cr_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            cr_optimizer.step()

            for p in critic.parameters():
                p.data.clamp_(-0.1, 0.1)
            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

        for cover, _ in tqdm(train_loader):
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            decoded = decoder.forward(generated)

            encoder_mse = mse_loss(generated, cover)
            decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
            decoder_acc = (decoded >= 0.0).eq(
                payload >= 0.5).sum().float() / payload.numel()
            generated_score = torch.mean(critic.forward(generated))

            en_de_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss +
             generated_score).backward()  # Why 100?
            en_de_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

        for cover, _ in tqdm(valid_loader):
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload)
            decoded = decoder.forward(generated)

            encoder_mse = mse_loss(generated, cover)
            decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
            decoder_acc = (decoded >= 0.0).eq(
                payload >= 0.5).sum().float() / payload.numel()
            generated_score = torch.mean(critic.forward(generated))
            cover_score = torch.mean(critic.forward(cover))

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(
                pytorch_ssim.ssim(cover, generated).item())
            metrics['val.psnr'].append(
                10 * torch.log10(4 / encoder_mse).item())
            metrics['val.bpp'].append(
                data_depth * (2 * decoder_acc.item() - 1))
        now = datetime.datetime.now()
        name = "EN_DE_%+.3f_%s.dat" % (cover_score.item(),
                                       now.strftime("%Y-%m-%d_%H:%M:%S"))
        fname = os.path.join('.', 'results/model', name)
        states = {
            'state_dict_critic': critic.state_dict(),
            'state_dict_encoder': encoder.state_dict(),
            'state_dict_decoder': decoder.state_dict(),
            'en_de_optimizer': en_de_optimizer.state_dict(),
            'cr_optimizer': cr_optimizer.state_dict(),
            'metrics': metrics,
            'train_epoch': ep,
            'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
        }
        torch.save(states, fname)
        plot('encoder_mse', ep, metrics['val.encoder_mse'], True)
        plot('decoder_loss', ep, metrics['val.decoder_loss'], True)
        plot('decoder_acc', ep, metrics['val.decoder_acc'], True)
        plot('cover_score', ep, metrics['val.cover_score'], True)
        plot('generated_score', ep, metrics['val.generated_score'], True)
        plot('ssim', ep, metrics['val.ssim'], True)
        plot('psnr', ep, metrics['val.psnr'], True)
        plot('bpp', ep, metrics['val.bpp'], True)


if __name__ == '__main__':
    for func in [
            lambda:os.mkdir(os.path.join('.', 'results')),
            lambda: os.mkdir(os.path.join('.', 'results/model')),
            lambda: os.mkdir(os.path.join('.', 'results/plots'))]:  # create directories
        try:
            func()
        except Exception as error:
            print(error)
            continue
    main()
