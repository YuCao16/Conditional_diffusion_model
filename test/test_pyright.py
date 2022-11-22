import contextlib
from typing import Dict

import torch
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


test = Test()
print(test.test())
