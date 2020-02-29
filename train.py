import gc
import os
import torch
from torch.optim import Adam
import torchvision
from torchvision import datasets, transforms
from encoder import BasicEncoder
from decoder import BasicDecoder
from critic import BasicCritic
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

def main():
	data_dir = ‘SteganoData’
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
	    							transforms.RandomCrop(360, pad_if_needed=True),
	    							transforms.ToTensor(),
	    							transforms.Normalize(mu, sigma)])

	train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

	valid_set = datasets.ImageFolder(os.path.join(data_dir, "validation"), transform=transform)
	valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=4, shuffle=False)

	encoder = BasicEncoder(data_depth, hidden_size)
	decoder = BasicDecoder(data_depth, hidden_size)
	critic = BasicCritic(hidden_size)
	cr_optimizer = Adam(critic.parameters(), lr=1e-4)
	de_optimizer = Adam(list(decoder.parameters())+list(encoder.parameters()), lr=1e-4) #Why add encoder parameters too?

	for ep in range(epochs):
		metrics = {field: list() for field in METRIC_FIELDS}
		for cover, _ in tqdm(train_loader):
			gc.collect()
			cover = cover.to(device)
			N, _, H, W = cover.size()
			payload = torch.zeros((N, data_depth, H, W), device).random_(0, 2) #sampled from the discrete uniform distribution over 0 to 2
			generated = encoder.forward(cover, payload)
			cover_score = torch.mean(critic.forward(cover))
			generated_score = torch.mean(critic.forward(generated))

			cr_optimizer.zero_grad()
			(cover_score - generated_score).backward(retain_graph=False)
			cr_optimizer.step()

			# for p in self.critic.parameters(): #What is this?
			#     p.data.clamp_(-0.1, 0.1)
			metrics['train.cover_score'].append(cover_score.item())
			metrics['train.generated_score'].append(generated_score.item())

		for cover, _ in tqdm(train_loader):
			gc.collect()
			cover = cover.to(device)
			N, _, H, W = cover.size()
			payload = torch.zeros((N, data_depth, H, W), device).random_(0, 2) #sampled from the discrete uniform distribution over 0 to 2
			generated = encoder.forward(cover, payload)
			decoded = decoder.forward(generated)

			encoder_mse = mse_loss(generated, cover)
			decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
			#decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
			generated_score = torch.mean(critic.forward(generated))

			de_optimizer.zero_grad()
			(100.0 * encoder_mse + decoder_loss + generated_score).backward() #Why 100?
			de_optimizer.step()


			metrics['train.encoder_mse'].append(encoder_mse.item())
			metrics['train.decoder_loss'].append(decoder_loss.item())
			#metrics['train.decoder_acc'].append(decoder_acc.item())

		for cover, _ in tqdm(valid_loader):
			gc.collect()
			cover = cover.to(device)
			N, _, H, W = cover.size()
			payload = torch.zeros((N, data_depth, H, W), device).random_(0, 2) #sampled from the discrete uniform distribution over 0 to 2
			generated = encoder.forward(cover, payload)
			decoded = decoder.forward(generated)

			encoder_mse = mse_loss(generated, cover)
			decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
			#decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
			generated_score = torch.mean(critic.forward(generated))
			cover_score = torch.mean(critic.forward(cover))

			metrics['val.encoder_mse'].append(encoder_mse.item())
			metrics['val.decoder_loss'].append(decoder_loss.item())
			metrics['val.decoder_acc'].append(decoder_acc.item())
			metrics['val.cover_score'].append(cover_score.item())
			metrics['val.generated_score'].append(generated_score.item())
			metrics['val.ssim'].append(ssim(cover, generated).item())
			metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
			metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

	print(train_loader.shape)

if __name__ == '__main__':
    main()