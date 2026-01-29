import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileConditionedBasisCoefficients(nn.Module):
    """Quantile-conditioned modulation for basis coefficients.

    Implements θ_k(τ) = θ_k^base · (1 + α_k · g(τ)) with a small learnable
    modulation scale. Initialized near identity to recover the standard model
    when α_k → 0.
    """

    def __init__(self, num_basis_functions: int, quantile_embed_dim: int = 8, modulation_scale_init: float = 0.001):
        super().__init__()
        self.num_basis = num_basis_functions

        self.quantile_encoder = nn.Sequential(
            nn.Linear(1, quantile_embed_dim),
            nn.SiLU(),
            nn.Linear(quantile_embed_dim, quantile_embed_dim),
            nn.SiLU(),
        )

        self.basis_modulation = nn.Linear(quantile_embed_dim, num_basis_functions)
        nn.init.zeros_(self.basis_modulation.weight)
        nn.init.zeros_(self.basis_modulation.bias)

        self.modulation_scale = nn.Parameter(torch.tensor(modulation_scale_init))

    def forward(self, base_coefficients: torch.Tensor, quantile_levels: torch.Tensor) -> torch.Tensor:
        """Apply quantile-dependent modulation.

        Args:
            base_coefficients: [B, Q, K] tensor of base coefficients.
            quantile_levels: [B, Q] or [Q] tensor with quantile levels τ.

        Returns:
            Tensor with the same shape as base_coefficients containing modulated coefficients.
        """

        if quantile_levels.dim() == 1:
            quantile_levels = quantile_levels.unsqueeze(0).expand(base_coefficients.size(0), -1)
        elif quantile_levels.size(0) == 1 and base_coefficients.size(0) > 1:
            quantile_levels = quantile_levels.expand(base_coefficients.size(0), -1)

        q_embed = self.quantile_encoder(quantile_levels.unsqueeze(-1))  # [B, Q, D]
        modulation = torch.tanh(self.basis_modulation(q_embed)) * self.modulation_scale  # [B, Q, K]
        
        return base_coefficients * (1 + modulation)

    @torch.no_grad()
    def get_modulation_analysis(self, quantile_levels: torch.Tensor) -> torch.Tensor:
        """Return per-quantile modulation factors for interpretability."""
        if quantile_levels.dim() == 1:
            quantile_levels = quantile_levels.unsqueeze(0)
        q_embed = self.quantile_encoder(quantile_levels.unsqueeze(-1))
        modulation = torch.tanh(self.basis_modulation(q_embed)) * self.modulation_scale
        return modulation


class NbeatsBlock(torch.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, size_in: int, size_out: int):
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.size_in = size_in
        self.size_out = size_out

        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        
        self.forward_projection = torch.nn.Linear(layer_width, size_out)
        self.backward_projection = torch.nn.Linear(layer_width, size_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.fc_layers:
            h = F.relu(layer(h))
        return self.forward_projection(h), self.backward_projection(h)
    
    
class NbeatsBlockConditioned(NbeatsBlock):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, size_in: int, size_out: int):
        
        super().__init__(num_layers=num_layers, layer_width=layer_width, size_in=size_in, size_out=size_out)
        
        self.condition_film = torch.nn.Linear(self.layer_width, 2*self.layer_width)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.fc_layers):
            h = F.relu(layer(h))
            if i == 0:
                condition_film = self.condition_film(condition)
                condition_offset, condition_delta = condition_film[..., :self.layer_width], condition_film[..., self.layer_width:]
                h = h * (1 + condition_delta) + condition_offset
            
        return self.forward_projection(h), self.backward_projection(h)


class NbeatsBlockQCBC(torch.nn.Module):
    """N-BEATS block with quantile-conditioned basis coefficients."""

    def __init__(self, num_layers: int, layer_width: int, size_in: int, size_out: int,
                 quantile_embed_dim: int = 8, modulation_scale_init: float = 0.001):
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.size_in = size_in
        self.size_out = size_out

        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)

        self.forward_projection = torch.nn.Linear(layer_width, size_out)
        self.backward_projection = torch.nn.Linear(layer_width, size_in)

        self.forward_qcbc = QuantileConditionedBasisCoefficients(
            num_basis_functions=size_out,
            quantile_embed_dim=quantile_embed_dim,
            modulation_scale_init=modulation_scale_init,
        )

    def forward(self, x: torch.Tensor, quantile_levels: torch.Tensor) -> torch.Tensor:
        # x: [B, Q, size_in], quantile_levels: [B, Q]
        h = x
        for layer in self.fc_layers:
            h = F.relu(layer(h))

        f_raw = self.forward_projection(h)
        b_raw = self.backward_projection(h)

        # Only modulate forward projection - preserve residual dynamics
        f = self.forward_qcbc(f_raw, quantile_levels)
        return f, b_raw
    
    
class NBEATS(torch.nn.Module):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, 
                 size_in: int, size_out: int, block_class: torch.nn.Module = NbeatsBlock):
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.size_in = size_in
        self.size_out = size_out
        self.num_blocks = num_blocks
        self.share = share
        
        self.blocks = [block_class(num_layers=num_layers, 
                                   layer_width=layer_width, 
                                   size_in=size_in, size_out=size_out)]
        if self.share:
            for i in range(self.num_blocks-1):
                self.blocks.append(self.blocks[0])
        else:
            for i in range(self.num_blocks-1):
                self.blocks.append(block_class(num_layers=num_layers, 
                                               layer_width=layer_width, 
                                               size_in=size_in, size_out=size_out))
        self.blocks = torch.nn.ModuleList(self.blocks)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backcast = x
        output = 0.0
        for block in self.blocks:
            f, b = block(backcast)
            output = output + f
            backcast = backcast - b
        return output


class NBEATSAQCAT(NBEATS):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, size_in: int, size_out: int, 
                 dropout: float = 0.0, quantile_embed_dim: int = 64, quantile_embed_num: int = 100):
        # size_in + 1, because one position for quantile
        super().__init__(num_blocks=num_blocks, num_layers=num_layers, 
                         layer_width=layer_width, share=share, 
                         size_in=size_in+1, size_out=size_out, block_class=NbeatsBlock)
        self.layer_width = layer_width

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input for hierarchical prediction.
        
        Args:
            x: Input tensor [B, T]
        Returns:
            Features tensor [B, layer_width]
        """
        # Use median quantile for feature extraction
        Q = 1
        q_median = torch.ones(x.shape[0], Q).to(x) * 0.5
        backcast = torch.cat([x.unsqueeze(1), q_median.unsqueeze(-1)], dim=-1)  # [B, 1, T+1]
        
        # Pass through FIRST block to extract features
        h = backcast.squeeze(1)  # [B, T+1]
        for layer in self.blocks[0].fc_layers:
            h = torch.nn.functional.relu(layer(h))  # [B, layer_width]
        
        # Return features from first block
        return h  # [B, layer_width]

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x history time series B x T
        # q quantile specification B x Q
        # output forward prediction H horizons and Q quantiles B x H x Q
        Q = q.shape[-1]
        backcast = torch.cat([torch.repeat_interleave(x[:, None], repeats=Q, dim=1, output_size=Q), q[..., None]], dim=-1) 

        output = 0.0
        for block in self.blocks:
            f, b = block(backcast)
            output = output + f
            backcast = backcast - b

        return output.transpose(-1, -2)


class NBEATSAQOUT(NBEATS):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, size_in: int, size_out: int):
        # size_in + 1, because one position for quantile
        super().__init__(num_blocks=num_blocks, num_layers=num_layers, 
                         layer_width=layer_width, share=share, 
                         size_in=size_in, size_out=size_out, block_class=NbeatsBlock)
        
        self.q_block = NbeatsBlockConditioned(layer_width=layer_width, size_in=size_in, 
                                              size_out=size_out, num_layers=num_layers)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x history time series B x T
        # q quantile specification B x Q
        # output forward prediction H horizons and Q quantiles B x H x Q

        backcast = x
        output = 0.0
        for block in self.blocks:
            f, b = block(backcast)
            output = output + f
            backcast = backcast - b

        Q = q.shape[-1]
        backcast = torch.repeat_interleave(backcast[:, None], repeats=Q, dim=1, output_size=Q)
        q = torch.repeat_interleave(q[..., None], repeats=self.layer_width, dim=-1, output_size=self.layer_width)
        
        f, b = self.q_block(backcast, condition=q)
        output = output[:, None] + f
        
        return output.transpose(-1, -2)


class NBEATSAQFILM(NBEATS):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, size_in: int, size_out: int):
        # size_in + 1, because one position for quantile
        super().__init__(num_blocks=num_blocks, num_layers=num_layers, 
                         layer_width=layer_width, share=share, 
                         size_in=size_in, size_out=size_out, block_class=NbeatsBlockConditioned)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x history time series B x T
        # q quantile specification B x Q
        # output forward prediction H horizons and Q quantiles B x H x Q

        Q = q.shape[-1]
        backcast = torch.repeat_interleave(x[:, None], repeats=Q, dim=1, output_size=Q)
        q = torch.repeat_interleave(q[..., None], repeats=self.layer_width, dim=-1, output_size=self.layer_width)

        output = 0.0
        for i, block in enumerate(self.blocks):
            f, b = block(x=backcast, condition=q)
            output = output + f
            backcast = backcast - b

        return output.transpose(-1, -2)


class NBEATSAQQCBC(torch.nn.Module):
    """Any-Quantile N-BEATS with quantile-conditioned basis coefficients."""

    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool,
                 size_in: int, size_out: int, quantile_embed_dim: int = 8,
                 modulation_scale_init: float = 0.001):
        super().__init__()
        self.num_blocks = num_blocks
        self.share = share
        self.layer_width = layer_width
        self.size_in = size_in + 1  # +1 for quantile concatenation
        self.size_out = size_out

        first_block = NbeatsBlockQCBC(
            num_layers=num_layers,
            layer_width=layer_width,
            size_in=self.size_in,
            size_out=size_out,
            quantile_embed_dim=quantile_embed_dim,
            modulation_scale_init=modulation_scale_init,
        )

        blocks = [first_block]
        if self.share:
            for _ in range(self.num_blocks - 1):
                blocks.append(first_block)
        else:
            for _ in range(self.num_blocks - 1):
                blocks.append(
                    NbeatsBlockQCBC(
                        num_layers=num_layers,
                        layer_width=layer_width,
                        size_in=self.size_in,
                        size_out=size_out,
                        quantile_embed_dim=quantile_embed_dim,
                        modulation_scale_init=modulation_scale_init,
                    )
                )

        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x: [B, T], q: [B, Q]
        Q = q.shape[-1]
        backcast = torch.cat([
            torch.repeat_interleave(x[:, None], repeats=Q, dim=1, output_size=Q),
            q[..., None],
        ], dim=-1)

        output = 0.0
        for block in self.blocks:
            f, b = block(backcast, quantile_levels=q)
            output = output + f
            backcast = backcast - b

        return output.transpose(-1, -2)
