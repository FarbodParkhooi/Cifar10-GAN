try:
    if __name__ == "__main__":
        from modules import dataset, utils, discriminator, generator
        from colorama import init, Fore, Back, Style
        import torchvision.utils as vutils
        import torch.optim as optim
        from os import makedirs
        from torch import nn
        import matplotlib
        import numpy
        import torch
        import time

        # Defining values
        configs = utils.Configs()
        dataloader = dataset.get_dataloader()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Torch device
        label_smoothing_labels = torch.ones(configs.batch_size, 1, device=device) * configs.label_smoothing_value  # label smoothing
        real_labels = torch.ones(configs.batch_size, 1, device=device)   # 1 labels for real images
        fake_labels = torch.zeros(configs.batch_size, 1, device=device)  # 0 labels for fake images
        fixed_noise = torch.randn(24, configs.G_latent_dim, device=device) # Fixed noise for creating same images
        start_time = 0
        begin_time = round(time.time())
        epoch_times = [0]
        average_func = lambda lst: sum(lst) / len(lst)

        # Related to libraries
        matplotlib.use('Agg') # preventing from crash
        import matplotlib.pyplot as plt # importing pyplot with new backend
        init() # init Colorama

        # Creating output directory
        makedirs(configs.output_directory, exist_ok=True)
        makedirs(f"{configs.output_directory}/{configs.output_images_directory}", exist_ok=True)
        makedirs(f"{configs.output_directory}/{configs.output_models_directory}", exist_ok=True)

        # Defining models
        G = generator.Generator(latent_dim=configs.G_latent_dim, projection_dim=configs.G_projection_dim).to(device)
        D = discriminator.Discriminator().to(device)

        # Defining loss function
        D_loss_real = lambda x: torch.clamp(1-x, min=0).mean()
        D_loss_fake = lambda x: torch.clamp(1+x, min=0).mean()
        G_loss = lambda x: -(x.mean())

        # Defining optimizer
        G_opt = optim.Adam(G.parameters(), lr=configs.G_learning_rate, betas=configs.betas)
        D_opt = optim.Adam(D.parameters(), lr=configs.D_learning_rate, betas=configs.betas)

        print(f"""{Fore.GREEN}{Style.BRIGHT}Starting training GAN model on CIFAR-10 dataset {Fore.WHITE}{Style.NORMAL}

        {Fore.BLUE}{Style.BRIGHT}-->   Generator Architecture {Fore.WHITE}{Style.NORMAL}:
        {G}

        {Fore.BLUE}{Style.BRIGHT}-->   Discriminator Architecture {Fore.WHITE}{Style.NORMAL}:
        {D}

        {Fore.BLUE}{Style.BRIGHT}-->   Generator Parameters{Fore.WHITE}{Style.NORMAL}:      {utils.count_parameters(G)}
        {Fore.BLUE}{Style.BRIGHT}-->   Discriminator Parameters{Fore.WHITE}{Style.NORMAL}:  {utils.count_parameters(D)} {Fore.WHITE}{Style.NORMAL}""")
        print(f"\n{Fore.GREEN}{Style.BRIGHT}                            ===- Using GPU -=== {Fore.WHITE}{Style.NORMAL}") if torch.cuda.is_available() else print(f"\n{Fore.YELLOW}{Style.BRIGHT}                            ===- Using CPU -=== {Fore.WHITE}{Style.NORMAL}")

        # lists for saving losses
        G_losses = []
        D_losses = []

        # Training loop
        for epoch_num in range(configs.epochs):
            start_time = time.time()
            for batch_idx, (real_images, _) in enumerate(dataloader):
                real_images = real_images + configs.real_images_noise * torch.randn_like(real_images)
                real_images = real_images.to(device)
                # Traning discriminator
                for _ in range(configs.train_D_per_epoch):
                    # Feed discriminator with real images
                    # Zero gradients
                    D_opt.zero_grad()

                    # Calculating loss on real images
                    output_real = D(real_images)
                    loss_D_real = D_loss_real(output_real)

                    # Feed discriminator with fake images
                    z = torch.randn(configs.batch_size, configs.G_latent_dim, device=device)
                    fake_images = G(z).detach()

                    # Calculating loss on fake images
                    output_fake = D(fake_images.to(device))
                    loss_D_fake = D_loss_fake(output_fake)

                    # Calculating total loss
                    loss_D = loss_D_real + loss_D_fake

                    # Training the discriminator
                    loss_D.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=configs.D_gradients_clipping_value)   # clip discriminator gradients
                    D_opt.step()

                # Training generator
                for _ in range(configs.train_G_per_epoch):
                    # Zero gradients
                    G_opt.zero_grad()

                    # Creating fake_images
                    z = torch.randn(configs.batch_size, configs.G_latent_dim, device=device)
                    fake_images = G(z)

                    # Feeding the discriminator and calculating loss
                    output = D(fake_images)
                    loss_G = G_loss(output)

                    # Training the model
                    loss_G.backward()
                    G_opt.step()

                # Adding losses 
                D_losses.append(loss_D.item())
                G_losses.append(loss_G.item())

                # Logging in terminal
                if batch_idx % configs.print_log_every_batch == 0 and batch_idx != 0:
                    print(f"{Fore.BLUE}{Style.NORMAL}Epoch {Fore.GREEN}{Style.BRIGHT}{epoch_num+1}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{configs.epochs} "
                        f"{Fore.BLUE}{Style.NORMAL}Batch {Fore.GREEN}{Style.BRIGHT}{batch_idx}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{len(dataloader)} "
                        f"{Fore.BLUE}{Style.NORMAL}D_loss: {Fore.GREEN}{Style.BRIGHT}{loss_D.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}G_loss: {Fore.GREEN}{Style.BRIGHT}{loss_G.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_real: {Fore.GREEN}{Style.BRIGHT}{output_real.mean().item():.3f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_fake: {Fore.GREEN}{Style.BRIGHT}{output_fake.mean().item():.3f}{Fore.WHITE}{Style.NORMAL} "
                        f"{Fore.BLUE}{Style.NORMAL}Total time: {Fore.GREEN}{Style.BRIGHT}{round((round(time.time()-begin_time))/60)}mins "
                        f"{Fore.BLUE}{Style.NORMAL}ETA: {Fore.GREEN}{Style.BRIGHT}{round((average_func(epoch_times)*(configs.epochs-epoch_num))/60)}mins")
                elif batch_idx == 0:
                    print(f"{Fore.BLUE}{Style.NORMAL}Epoch {Fore.GREEN}{Style.BRIGHT}{epoch_num+1}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{configs.epochs} "
                        f"{Fore.BLUE}{Style.NORMAL}Batch {Fore.GREEN}{Style.BRIGHT}000{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{len(dataloader)} "
                        f"{Fore.BLUE}{Style.NORMAL}D_loss: {Fore.GREEN}{Style.BRIGHT}{loss_D.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}G_loss: {Fore.GREEN}{Style.BRIGHT}{loss_G.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_real: {Fore.GREEN}{Style.BRIGHT}{output_real.mean().item():.3f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_fake: {Fore.GREEN}{Style.BRIGHT}{output_fake.mean().item():.3f}{Fore.WHITE}{Style.NORMAL} "
                        f"{Fore.BLUE}{Style.NORMAL}Total time: {Fore.GREEN}{Style.BRIGHT}{round((round(time.time()-begin_time))/60)}mins "
                        f"{Fore.BLUE}{Style.NORMAL}ETA: {Fore.GREEN}{Style.BRIGHT}{round((average_func(epoch_times)*(configs.epochs-epoch_num))/60)}mins")

            # Logging the status after each epoch
            print(f"{Fore.BLUE}{Style.NORMAL}Epoch {Fore.GREEN}{Style.BRIGHT}{epoch_num+1}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{configs.epochs} "
                    f"{Fore.BLUE}{Style.NORMAL}Batch {Fore.GREEN}{Style.BRIGHT}{len(dataloader)}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{len(dataloader)} "
                    f"{Fore.BLUE}{Style.NORMAL}D_loss: {Fore.GREEN}{Style.BRIGHT}{loss_D.item():.4f} "
                    f"{Fore.BLUE}{Style.NORMAL}G_loss: {Fore.GREEN}{Style.BRIGHT}{loss_G.item():.4f} "
                    f"{Fore.BLUE}{Style.NORMAL}D_real: {Fore.GREEN}{Style.BRIGHT}{output_real.mean().item():.3f} "
                    f"{Fore.BLUE}{Style.NORMAL}D_fake: {Fore.GREEN}{Style.BRIGHT}{output_fake.mean().item():.3f}{Fore.WHITE}{Style.NORMAL} "
                    f"{Fore.BLUE}{Style.NORMAL}Total time: {Fore.GREEN}{Style.BRIGHT}{round((round(time.time()-begin_time))/60)}mins "
                    f"{Fore.BLUE}{Style.NORMAL}ETA: {Fore.GREEN}{Style.BRIGHT}{round((average_func(epoch_times)*(configs.epochs-epoch_num))/60)}mins")

            # Creating same image after each epoch
            # Set generator to evaluation mode
            G.eval()
            with torch.no_grad():
                # Generate images from fixed noise
                generated_images = G(fixed_noise).cpu()
            G.train()  # Back to training mode
            
            # Create image grid (8 images per row)
            grid = vutils.make_grid(generated_images, nrow=8, normalize=True)
            
            # Convert to numpy and display/save
            plt.figure(figsize=(10, 10))
            plt.imshow(numpy.transpose(grid, (1, 2, 0)))
            plt.axis('off')
            plt.title(f"Epoch {epoch_num+1}")
            plt.savefig(f"{configs.output_directory}/{configs.output_images_directory}/epoch_{epoch_num+1:03d}.png", bbox_inches='tight')
            plt.close()

            print(f"{Fore.MAGENTA}{Style.BRIGHT}Epoch {epoch_num+1} complete. Images saved to {configs.output_directory}/{configs.output_images_directory}/epoch_{epoch_num+1:03d}.png{Fore.WHITE}{Style.NORMAL}")

            if (epoch_num+1) % configs.save_model_every_epoch == 0:
                torch.save(G.state_dict(), f"{configs.output_directory}/{configs.output_models_directory}/cifar10_GAN_epoch_{epoch_num+1}.pth")
                print(f"{Fore.MAGENTA}{Style.BRIGHT}Model saved to {configs.output_directory}/{configs.output_models_directory}/epoch_{epoch_num+1:03d}.png at epoch {epoch_num+1:03d}{Fore.WHITE}{Style.NORMAL}")

            # Calculating the time spent on this epoch
            epoch_times.append(round(time.time() - start_time))

            # Removing 0 from epoch_times
            if (epoch_num+1) == 2:
                del epoch_times[0]

        # Plotting training process
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="Generator Loss")
        plt.plot(D_losses, label="Discriminator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{configs.output_directory}/loss_plot.png", bbox_inches='tight')
        plt.show()

        print(f"{Fore.BLUE}{Style.BRIGHT}Final Generator Loss{Fore.WHITE}{Style.NORMAL}:      {G_losses[-1]:.4f}")
        print(f"{Fore.BLUE}{Style.BRIGHT}Final Discriminator Loss{Fore.WHITE}{Style.NORMAL}:  {D_losses[-1]:.4f}")

        # Saving model
        torch.save(G.state_dict(), f"{configs.output_directory}/{configs.output_models_directory}/cifar10_GAN.pth")
        print(f"\n{Fore.GREEN}{Style.NORMAL}Generator saved to {Fore.BLUE}{Style.BRIGHT}'{configs.output_directory}/{configs.output_models_directory}/cifar10_GAN.pth'")

except KeyboardInterrupt:
    print("Going out...")
