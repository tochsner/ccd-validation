import torch
from torch import nn
from src.ml.modeling.layers.flow_layers import FlowLayer


class InvertibleAffine(FlowLayer):
    """
    Invertible affine transformation without shift, i.e. one-dimensional
    version of the invertible 1x1 convolutions
    """

    def __init__(self, num_channels):
        """Constructor

        Args:
          num_channels: Number of channels of the data
          use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        
        Q, _ = torch.linalg.qr(torch.randn(self.num_channels, self.num_channels))
            
        P, L, U = torch.lu_unpack(*Q.lu())
        self.register_buffer("P", P)  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        S = U.diag()  # "crop out" the diagonal to its own parameter
        self.register_buffer("sign_S", torch.sign(S))
        self.log_S = nn.Parameter(torch.log(torch.abs(S)))
        self.U = nn.Parameter(
            torch.triu(U, diagonal=1)
        )  # "crop out" diagonal, stored in S
        self.register_buffer("eye", torch.diag(torch.ones(self.num_channels)))

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(
            self.sign_S * torch.exp(self.log_S)
        )
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

    def inverse(self, z, **kwargs):
        W = self._assemble_W(inverse=True)
        log_det = -self.log_S
        
        z_ = z @ W

        return {
            "z": z_,
            "log_dj": log_det,
        }

    def forward(self, z, **kwargs):
        W = self._assemble_W()
        log_det = self.log_S
        
        z_ = z @ W

        return {
            "z": z_,
            "log_dj": log_det,
        }
