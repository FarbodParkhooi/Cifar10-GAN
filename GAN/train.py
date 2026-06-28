try:
    if __name__ == "__main__":
        from modules import dataset, discriminator, generator
        from colorama import init, Fore, Back, Style
        import torchvision.utils as vutils
        from datetime import datetime
        import torch.optim as optim
        from modules import configs
        from os import makedirs
        from torch import nn
        import matplotlib
        import numpy
        import torch
        import time
        import json
        import warnings

        # Defining values
        cfg = configs.Configs()
        dataloader = dataset.get_dataloader()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Torch device
        fixed_noise = torch.randn(32, cfg.G_latent_dim, device=device) # Fixed noise for creating same images
        start_time = 0
        begin_time = round(time.time())
        epoch_times = [0]
        average_func = lambda lst: sum(lst) / len(lst)
        start_epoch = 0
        

        # Related to libraries
        matplotlib.use('Agg') # preventing from crash
        import matplotlib.pyplot as plt # importing pyplot with new backend
        init() # init Colorama
        torch.backends.cudnn.benchmark = cfg.enable_cudnn  # CudNN configs 
        warnings.filterwarnings("ignore")  # Disabling warnings

        # Creating output directory
        makedirs(cfg.output_directory, exist_ok=True)
        makedirs(f"{cfg.output_directory}/{cfg.output_images_directory}", exist_ok=True)
        makedirs(f"{cfg.output_directory}/{cfg.output_models_directory}", exist_ok=True)

        # Defining models
        G = generator.Generator(latent_dim=cfg.G_latent_dim, projection_dim=cfg.G_projection_dim).to(device)
        D = discriminator.Discriminator().to(device)

        # Defining loss function
        D_loss_real = lambda x: torch.clamp(1-x, min=0).mean()
        D_loss_fake = lambda x: torch.clamp(1+x, min=0).mean()
        G_loss = lambda x: -(x.mean())

        # Defining optimizer
        G_opt = optim.Adam(G.parameters(), lr=cfg.G_learning_rate, betas=cfg.betas)
        D_opt = optim.Adam(D.parameters(), lr=cfg.D_learning_rate, betas=cfg.betas)

        # Defining MultiStepLR
        D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_opt, milestones=cfg.reduce_epochs, gamma=cfg.reduce_lr)

        # Load from checkpoint
        if cfg.load_from_checkpoint:
            try:
                print(f"{Fore.GREEN}{Style.NORMAL}Trying to load model. ", end="")
                G.load_state_dict(torch.load(f"{cfg.checkpoint_directory}/generator.pth"))
                print(f"{Style.BRIGHT}Loaded generator. ", end="")
                D.load_state_dict(torch.load(f"{cfg.checkpoint_directory}/discriminator.pth"))
                print(f"Loaded discriminator. ", end="")
                G_opt.load_state_dict(torch.load(f"{cfg.checkpoint_directory}/generator_optimizer.pth"))
                print(f"Loaded generator's optimizer. ", end="")
                D_opt.load_state_dict(torch.load(f"{cfg.checkpoint_directory}/discriminator_optimizer.pth"))
                print(f"Loaded discriminator's optimizer. ", end="")
                D_scheduler.load_state_dict(torch.load(f"{cfg.checkpoint_directory}/lr_scheduler.pth")) 
                print(f"Loaded discriminator's lr scheduler. ", end="")
                with open(f"{cfg.checkpoint_directory}/status.json") as file:  
                    start_epoch = json.load(file)["epoch_num"]+1
                print(f"Loaded epoch number. ", end="")
                print("Loaded model successfuly.")
            except Exception as e:
                print(f"{Fore.RED}{Back.LIGHTBLACK_EX}{Style.BRIGHT}ERROR WHILE LOADING FROM CHECKPOING. CHECK FOR FILES(Error: {e}){Fore.WHITE}{Back.BLACK}{Style.NORMAL}")

        # Logging initial information
        print(f"""{Fore.GREEN}{Style.BRIGHT}Starting training GAN model on CIFAR-10 dataset {Fore.WHITE}{Style.NORMAL}

        {Fore.BLUE}{Style.BRIGHT}-->   Generator Architecture {Fore.WHITE}{Style.NORMAL}:
        {G}

        {Fore.BLUE}{Style.BRIGHT}-->   Discriminator Architecture {Fore.WHITE}{Style.NORMAL}:
        {D}

        {Fore.BLUE}{Style.BRIGHT}-->   Generator Parameters{Fore.WHITE}{Style.NORMAL}:      {configs.count_parameters(G)}
        {Fore.BLUE}{Style.BRIGHT}-->   Discriminator Parameters{Fore.WHITE}{Style.NORMAL}:  {configs.count_parameters(D)} {Fore.WHITE}{Style.NORMAL}""")
        print(f"\n{Fore.GREEN}{Style.BRIGHT}                            ===- Using GPU -=== {Fore.WHITE}{Style.NORMAL}") if torch.cuda.is_available() else print(f"\n{Fore.YELLOW}{Style.BRIGHT}                            ===- Using CPU -=== {Fore.WHITE}{Style.NORMAL}")

        # lists for saving losses
        G_losses = []
        D_losses = []

        # Training loop
        for epoch_num in range(start_epoch, cfg.epochs):
            start_time = time.time()
            for batch_idx, (real_images, _) in enumerate(dataloader):
                real_images = real_images + cfg.real_images_noise * torch.randn_like(real_images)
                real_images = real_images.to(device)
                # Traning discriminator
                for _ in range(cfg.train_D_per_epoch):
                    # Feed discriminator with real images
                    # Zero gradients
                    D_opt.zero_grad()

                    # Computing loss on real images
                    output_real = D(real_images)
                    loss_D_real = D_loss_real(output_real)

                    # Feed discriminator with fake images
                    z = torch.randn(cfg.batch_size, cfg.G_latent_dim, device=device)
                    fake_images = G(z).detach()

                    # Computing loss on fake images
                    output_fake = D(fake_images.to(device))
                    loss_D_fake = D_loss_fake(output_fake)

                    # Gradient Penalty
                    epsilon = torch.rand(cfg.batch_size, 1, 1, 1, device=device)
                    interpolated_images = epsilon*real_images+(1-epsilon)*fake_images
                    interpolated_images = interpolated_images.requires_grad_(True)
                    output_interpolated = D(interpolated_images)
                    # Computing gradient 
                    gradients = torch.autograd.grad(
                        outputs=output_interpolated.sum(),
                        inputs=interpolated_images,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    # Computing gradient norm
                    grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
                    # Computing penalty
                    penalty = ((grad_norm - 1.0) ** 2).mean()

                    # Computing total loss
                    loss_D = loss_D_real + loss_D_fake + cfg.penalty_effect*penalty
                    
                    # Training the discriminator
                    loss_D.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=cfg.D_gradients_clipping_value)   # clip discriminator gradients
                    D_opt.step()

                # Training generator
                for _ in range(cfg.train_G_per_epoch):
                    # Zero gradients
                    G_opt.zero_grad()

                    # Creating fake_images
                    z = torch.randn(cfg.batch_size, cfg.G_latent_dim, device=device)
                    fake_images = G(z)

                    # Feeding the discriminator and Computing loss
                    output = D(fake_images)
                    loss_G = G_loss(output)

                    # Training the model
                    loss_G.backward()
                    G_opt.step()

                # Adding losses 
                D_losses.append(loss_D.item())
                G_losses.append(loss_G.item())

                # Logging in terminal
                if batch_idx % cfg.print_log_every_batch == 0 and batch_idx != 0:
                    print(f"{Fore.BLUE}{Style.NORMAL}Epoch {Fore.GREEN}{Style.BRIGHT}{epoch_num+1}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{cfg.epochs} "
                        f"{Fore.BLUE}{Style.NORMAL}Batch {Fore.GREEN}{Style.BRIGHT}{batch_idx}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{len(dataloader)} "
                        f"{Fore.BLUE}{Style.NORMAL}D_loss: {Fore.GREEN}{Style.BRIGHT}{loss_D.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}G_loss: {Fore.GREEN}{Style.BRIGHT}{loss_G.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_real: {Fore.GREEN}{Style.BRIGHT}{output_real.mean().item():.3f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_fake: {Fore.GREEN}{Style.BRIGHT}{output_fake.mean().item():.3f}{Fore.WHITE}{Style.NORMAL} "
                        f"{Fore.BLUE}{Style.NORMAL}Total time: {Fore.GREEN}{Style.BRIGHT}{round((round(time.time()-begin_time))/60)}mins "
                        f"{Fore.BLUE}{Style.NORMAL}ETA: {Fore.GREEN}{Style.BRIGHT}{round((average_func(epoch_times)*(cfg.epochs-epoch_num))/60)}mins")
                elif batch_idx == 0:
                    print(f"{Fore.BLUE}{Style.NORMAL}Epoch {Fore.GREEN}{Style.BRIGHT}{epoch_num+1}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{cfg.epochs} "
                        f"{Fore.BLUE}{Style.NORMAL}Batch {Fore.GREEN}{Style.BRIGHT}000{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{len(dataloader)} "
                        f"{Fore.BLUE}{Style.NORMAL}D_loss: {Fore.GREEN}{Style.BRIGHT}{loss_D.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}G_loss: {Fore.GREEN}{Style.BRIGHT}{loss_G.item():.4f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_real: {Fore.GREEN}{Style.BRIGHT}{output_real.mean().item():.3f} "
                        f"{Fore.BLUE}{Style.NORMAL}D_fake: {Fore.GREEN}{Style.BRIGHT}{output_fake.mean().item():.3f}{Fore.WHITE}{Style.NORMAL} "
                        f"{Fore.BLUE}{Style.NORMAL}Total time: {Fore.GREEN}{Style.BRIGHT}{round((round(time.time()-begin_time))/60)}mins "
                        f"{Fore.BLUE}{Style.NORMAL}ETA: {Fore.GREEN}{Style.BRIGHT}{round((average_func(epoch_times)*(cfg.epochs-epoch_num))/60)}mins")

            # Logging the status after each epoch
            print(f"{Fore.BLUE}{Style.NORMAL}Epoch {Fore.GREEN}{Style.BRIGHT}{epoch_num+1}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{cfg.epochs} "
                    f"{Fore.BLUE}{Style.NORMAL}Batch {Fore.GREEN}{Style.BRIGHT}{len(dataloader)}{Fore.WHITE}{Style.NORMAL}/{Fore.GREEN}{Style.BRIGHT}{len(dataloader)} "
                    f"{Fore.BLUE}{Style.NORMAL}D_loss: {Fore.GREEN}{Style.BRIGHT}{loss_D.item():.4f} "
                    f"{Fore.BLUE}{Style.NORMAL}G_loss: {Fore.GREEN}{Style.BRIGHT}{loss_G.item():.4f} "
                    f"{Fore.BLUE}{Style.NORMAL}D_real: {Fore.GREEN}{Style.BRIGHT}{output_real.mean().item():.3f} "
                    f"{Fore.BLUE}{Style.NORMAL}D_fake: {Fore.GREEN}{Style.BRIGHT}{output_fake.mean().item():.3f}{Fore.WHITE}{Style.NORMAL} "
                    f"{Fore.BLUE}{Style.NORMAL}Total time: {Fore.GREEN}{Style.BRIGHT}{round((round(time.time()-begin_time))/60)}mins "
                    f"{Fore.BLUE}{Style.NORMAL}ETA: {Fore.GREEN}{Style.BRIGHT}{round((average_func(epoch_times)*(cfg.epochs-epoch_num))/60)}mins")

            # Creating same image after each epoch
            if (epoch_num+1) % cfg.save_sample_image_every_epoch == 0:
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
                print(f"{Fore.MAGENTA}{Style.BRIGHT}Images saved to {configs.output_directory}/{configs.output_images_directory}/epoch_{epoch_num+1:03d}.png{Fore.WHITE}{Style.NORMAL}")
                plt.savefig(f"{cfg.output_directory}/{cfg.output_images_directory}/epoch_{epoch_num:03d}.png", bbox_inches='tight')
                plt.close()
                print(f"{Fore.MAGENTA}{Style.BRIGHT}Images saved to {cfg.output_directory}/{cfg.output_images_directory}/epoch_{epoch_num:03d}.png{Fore.WHITE}{Style.NORMAL}")

            # Saving checkpoints every save_checkpoints_every_epoch epoch
            if (epoch_num+1) % cfg.save_checkpoints_every_epoch == 0:
                makedirs(f"{cfg.output_directory}/{cfg.output_models_directory}/epoch_{epoch_num+1:03d}/", exist_ok=True)
                # Saving generator
                torch.save(G.state_dict(), f"{cfg.output_directory}/{cfg.output_models_directory}/epoch_{epoch_num+1:03d}/generator.pth")
                # Saving discriminator
                torch.save(D.state_dict(), f"{cfg.output_directory}/{cfg.output_models_directory}/epoch_{epoch_num+1:03d}/discriminator.pth")
                # Saving generator's optimizer
                torch.save(G_opt.state_dict(), f"{cfg.output_directory}/{cfg.output_models_directory}/epoch_{epoch_num+1:03d}/generator_optimizer.pth")
                # Saving discriminator's optimizer
                torch.save(D_opt.state_dict(), f"{cfg.output_directory}/{cfg.output_models_directory}/epoch_{epoch_num+1:03d}/discriminator_optimizer.pth")
                # Saving discriminator's learning rate scheduler
                torch.save(D_scheduler.state_dict(), f"{cfg.output_directory}/{cfg.output_models_directory}/epoch_{epoch_num+1:03d}/lr_scheduler.pth")                
                # Saving epoch number
                with open(f"{cfg.output_directory}/{cfg.output_models_directory}/epoch_{epoch_num+1:03d}/status.json", "w") as file:
                    data = {"epoch_num" : epoch_num}
                    json.dump(data, file)
                print(f"{Fore.MAGENTA}{Style.BRIGHT}Checkpoint saved to {cfg.output_directory}/{cfg.output_models_directory}/ at epoch {epoch_num+1:03d}{Fore.WHITE}{Style.NORMAL}")

            # Removing 0 from epoch_times
            if (epoch_num+1) == 2:   del epoch_times[0]

            # Stepping for LR scheduler
            D_scheduler.step()

            # Computing the time spent on this epoch
            epoch_times.append(round(time.time() - start_time))

            # Finishing the epoch
            print(f"{Fore.MAGENTA}{Style.BRIGHT}======================================{Fore.WHITE}{Style.NORMAL}")

        # Plotting training process
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="Generator Loss")
        plt.plot(D_losses, label="Discriminator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{cfg.output_directory}/loss_plot.png", bbox_inches='tight')
        plt.show()
        
        # Saving model
        torch.save(G.state_dict(), f"{cfg.output_directory}/{cfg.output_models_directory}/cifar10_GAN.pth")
        print(f"\n{Fore.GREEN}{Style.NORMAL}Generator saved to {Fore.BLUE}{Style.BRIGHT}'{cfg.output_directory}/{cfg.output_models_directory}/cifar10_GAN.pth'")

        # Logging plot address
        print(f"{Fore.GREEN}{Style.BRIGHT}Training process plott saved to '{cfg.output_directory}/loss_plot.png'{Fore.WHITE}{Style.NORMAL}")

        # Logging models' losses
        print(f"{Fore.BLUE}{Style.BRIGHT}Final Generator Loss{Fore.WHITE}{Style.NORMAL}:      {G_losses[-1]:.4f}{Fore.WHITE}{Style.NORMAL}")
        print(f"{Fore.BLUE}{Style.BRIGHT}Final Discriminator Loss{Fore.WHITE}{Style.NORMAL}:  {D_losses[-1]:.4f}{Fore.WHITE}{Style.NORMAL}")

        # Logging last data
        print(f"{Fore.GREEN}{Style.BRIGHT}\n\nTraining Completed Successfuly!\n\n{Fore.WHITE}{Style.NORMAL}")
        print(f"{Fore.BLUE}       Finshed training at {datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}")


except KeyboardInterrupt:
    print("Going out...")
