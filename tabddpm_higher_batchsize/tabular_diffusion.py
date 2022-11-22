from typing import Dict, Tuple, Union

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader

# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from tabular_dataset import Dataset


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        """
        standard ResNet style convolutional block
        """
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            out = x + x2 if self.same_channels else x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            return self.conv2(x1)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        """
        process and downscale the image feature maps
        """
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool1d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        """
        process and upscale the image feature maps
        """
        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int) -> None:
        super(EmbedFC, self).__init__()
        """
        generic one layer FC NN for embedding things  
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels: int, n_feat: int = 256) -> None:
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool1d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose1d(
                2 * n_feat, 2 * n_feat, 7, 7
            ),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv1d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv1d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed context, time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(
        self,
        nn_model: nn.Module,
        betas: Tuple[float, ...],
        n_T: int,
        device: str,
        drop_prob: float = 0.1,
    ) -> None:
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        this method is used in training, so samples t and noise randomly
        """

        # t ~ Uniform(0, n_T)
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        # x with t embedding
        assert isinstance(self.sqrtab, torch.Tensor)
        assert isinstance(self.sqrtmab, torch.Tensor)
        x_t = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * noise

        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T))

    def sample(
        self, n_sample: int, size: Union[torch.Size, Tuple], device: str
    ) -> torch.Tensor:

        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)

        for i in range(self.n_T, 0, -1):
            print(f"sampleing timestep {i}", end="\r")

            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is)
            x_i = x_i[:n_sample]

            assert isinstance(self.mab_over_sqrtmab, torch.Tensor)
            assert isinstance(self.oneover_sqrta, torch.Tensor)
            assert isinstance(self.sqrt_beta_t, torch.Tensor)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def train_tabular() -> None:

    # hardcoding these here
    dataset_path = "./higgs.csv"
    n_epoch = 200
    batch_size = 128
    n_T = 600
    device = "cuda:0"
    n_feat = 256
    lrate = 1e-4
    save_model = True
    save_dir = "./data/tabular_diffusion_outputs/"

    ddpm = DDPM(
        nn_model=ContextUnet(1, n_feat),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
    )
    ddpm.to(device)

    dataset = Dataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.SGD(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None

        for x in pbar:
            optim.zero_grad()
            x = x.to(device, dtype=torch.float)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            n_sample = 100
            x_gen = ddpm.sample(n_sample, (1, 28), device).reshape((n_sample, -1))
            pd.DataFrame(x_gen.detach().cpu().numpy()).to_csv(
                f"{save_dir}tabular_ep{ep}.csv", header=False, index=False
            )
            print(f"saved tabular at {save_dir}" + f"tabular_ep{ep}.csv")

        if save_model and ep == int(n_epoch - 1):
            torch.save(ddpm.state_dict(), f"{save_dir}model_{ep}.pth")
            print(f"saved model at {save_dir}" + f"model_{ep}.pth")


if __name__ == "__main__":
    train_tabular()


# x: torch.Tensor = torch.ones((256, 1, 28))
# t: torch.Tensor = torch.ones((256))
#
# model_unet = ContextUnet(1, 256)
# model = DDPM(model_unet, betas=(1e-4, 0.02), n_T=400, device="cpu")
# y = model.sample(10, (1, 28), "cpu")
# print(y[0])
# y = model(x)
