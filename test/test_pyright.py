import contextlib
from typing import Dict, List, Tuple, Union

import torch
import numpy as np
from torch import nn


def foo() -> Dict[str, torch.Tensor]:
    alpha_t = torch.tensor([1.0, 1.5])
    return {"alpha_t": alpha_t}


class Test(nn.Module):
    def __init__(self) -> None:
        super(Test, self).__init__()

        k: str = ""
        for k, v in foo().items():
            self.register_buffer(k, v)
        with contextlib.suppress(ValueError):
            print(k)

    def test(self) -> torch.Tensor:
        print(self.alpha_t)
        assert isinstance(self.alpha_t, torch.Tensor)
        return self.alpha_t[0]


class DDPM(nn.Module):
    def __init__(
        self,
        nn_model: nn.Module,
        n_T: int,
        device: str,
        drop_prob: float = 0.1,
    ) -> None:
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in foo().items():
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
    ) -> Tuple[torch.Tensor, np.ndarray]:

        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)
        x_i_store = []  # keep track of generated steps

        for i in range(self.n_T, 0, -1):
            print(f"sampleing timestep {i}", end="\r")

            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1)

            # split predictions and compute weighting
            x_i = x_i[:n_sample]

            assert isinstance(self.oneover_sqrta, torch.Tensor)
            x_i = self.oneover_sqrta[i]
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


test = Test()
print(test.test())
