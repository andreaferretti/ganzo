import torch


class StandardGame:
    def __init__(self, options, generator, discriminator, loss):
        self.device = torch.device(options.device)
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.generator_iterations = options.generator_iterations
        self.discriminator_iterations = options.discriminator_iterations
        self.generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=options.generator_lr,
            betas=(options.beta1, options.beta2)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=options.discriminator_lr,
            betas=(options.beta1, options.beta2)
        )

    def run_epoch(self, dataloader, noiseloader):
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        for _ in range(self.generator_iterations):
            self.generator.zero_grad()
            noise = noiseloader.next().to(self.device)
            noise.requires_grad_(True)
            fake_data = self.generator(noise)
            generator_loss = self.discriminator(fake_data).mean()
            generator_loss.backward()

            self.generator_optimizer.step()

        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        for _ in range(self.discriminator_iterations):
            self.discriminator.zero_grad()
            noise = noiseloader.next().to(self.device)
            fake_data = self.generator(noise).detach()
            real_data, labels = dataloader.next()
            real_data = real_data.to(self.device)
            labels = labels.to(self.device)
            discriminator_loss = self.loss.run(real_data, fake_data, labels)
            discriminator_loss.backward()

            self.discriminator_optimizer.step()

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }

class Game:
    @staticmethod
    def from_options(options, generator, discriminator, loss):
        return StandardGame(options, generator, discriminator, loss)

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('training options')
        group.add_argument('--generator-iterations', type=int, default=1, help='number of iterations for the generator')
        group.add_argument('--discriminator-iterations', type=int, default=5, help='number of iterations for the discriminator')
        group.add_argument('--generator-lr', type=float, default=1e-4, help='learning rate for the generator')
        group.add_argument('--discriminator-lr', type=float, default=1e-4, help='learning rate for the discriminator')
        group.add_argument('--beta1', type=float, default=0, help='first beta')
        group.add_argument('--beta2', type=float, default=0.9, help='second beta')