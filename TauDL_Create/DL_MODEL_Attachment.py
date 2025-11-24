import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import math
from tqdm import tqdm
import numpy as np
from typing import Tuple, Optional, Dict, Any

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (B, T, D)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, hidden_size)
        residual = x
        
        # Self-Attention + Residual + Norm
        x, _ = self.attention(x, x, x)  # Q=K=V=x
        x = self.norm1(residual + self.dropout(x))
        
        # Feed-Forward + Residual + Norm
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        
        return x


def get_sinusoidal_position_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encoding table
    :param max_len: Maximum sequence length (number of vertical levels)
    :param d_model: Embedding dimension (hidden_size)
    :return: Positional encoding of shape (1, max_len, d_model)
    """
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
    
    pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    
    return pe  # Register as buffer; no gradients needed

class ResChannelAttention(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=3, output_size=15, num_heads=4, max_vertical_levels=100):
        super(ResChannelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.max_vertical_levels = max_vertical_levels

        # 1. Input projection
        self.linear = nn.Linear(input_size, hidden_size)

        # 2. Multiple Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_size, num_heads=num_heads) 
            for _ in range(num_layers)
        ])

        # 3. Final output layer (with residual connection)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size + input_size),
            nn.Linear(hidden_size + input_size, output_size)
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()

        # 4. Register sinusoidal positional encoding as buffer (no gradient update)
        self.register_buffer(
            'positional_encoding',
            get_sinusoidal_position_encoding(max_vertical_levels, hidden_size)
        )

    def forward(self, x):
        """
        x: (B, T, input_size)  T is the number of vertical levels (sequence length)
        """
        B, T, _ = x.shape

        # Freeze the 0th and 2nd variables (e.g., T, q, O3, p)
        # x_const1 = x[:, :, 0].unsqueeze(-1).detach()  # e.g., temperature
        x_var1   = x[:, :, 0].unsqueeze(-1)           # e.g., humidity
        x_var2   = x[:, :, 1].unsqueeze(-1)           # e.g., temperature
        x_const3 = x[:, :, 2].unsqueeze(-1).detach()  # e.g., zenith angle

        # Reconstruct input
        x = torch.cat([x_var1, x_var2, x_const3], dim=-1)  # (B, T, 3)

        skip = x  # Residual connection: original input

        # Project to hidden space
        x = self.linear(x)  # (B, T, hidden_size)

        # Add positional encoding (use first T positions only)
        # positional_encoding: (1, max_len, hidden_size)
        x = x + self.positional_encoding[:, :T, :]  # (B, T, hidden_size)

        # Pass through multiple attention blocks
        for block in self.attention_blocks:
            x = block(x)  # (B, T, hidden_size)

        x = self.norm(x)
        x = self.relu(x)

        # Concatenate with original input (residual)
        x = torch.cat([skip, x], dim=2)  # (B, T, 3 + hidden_size)

        return self.fc(x)  # (B, T, output_size)
    
class BiLSTMBlock(nn.Module):
    """
    A bidirectional LSTM block with residual connection and layer normalization.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the BiLSTM block.
        
        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden dimension for LSTM.
            output_size (int): Output feature dimension.
        """
        super(BiLSTMBlock, self).__init__()
        # Bidirectional LSTM: outputs concatenated forward and backward states
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True  # Input and output in (batch, seq_len, features) format
        )
        # Final linear layer: includes residual connection (input_size added)
        self.fc = nn.Linear(2 * hidden_size + input_size, output_size)
        # Layer normalization applied to LSTM output
        self.norm = nn.LayerNorm(2 * hidden_size)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, input_size)
        
        Returns:
            Tensor: Output of shape (B, T, output_size)
        """
        skip = x  # Store input for residual connection
        # LSTM processes the sequence; output shape: (B, T, 2 * hidden_size)
        x, _ = self.lstm(x)
        x = self.norm(x)  # Normalize LSTM output
        # Concatenate residual (original input) with LSTM output before ReLU and FC
        x = torch.cat([skip, x], dim=2)  # (B, T, input_size + 2*hidden_size)
        out = self.fc(self.relu(x))  # Apply ReLU and project to output_size
        return out


class ResChannelLSTM(nn.Module):
    """
    Residual Channel LSTM model that processes vertical atmospheric profiles.
    Uses frozen variables and deep BiLSTM blocks with residual connections.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the ResChannelLSTM model.
        
        Args:
            input_size (int): Number of input features (e.g., 3 after freezing).
            hidden_size (int): Dimension of hidden representations.
            num_layers (int): Number of stacked BiLSTM blocks.
            output_size (int): Number of output features (e.g., 15 for output channels).
        """
        super(ResChannelLSTM, self).__init__()
        # Linear projection from input space to hidden space
        self.linear = nn.Linear(input_size, hidden_size)
        
        # Stack of BiLSTM blocks with residual connections
        self.reslstm = nn.ModuleList([
            BiLSTMBlock(hidden_size, hidden_size, hidden_size) 
            for _ in range(num_layers)
        ])
        
        # Final output layer with LayerNorm and linear projection
        # Includes residual connection from original input
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size + input_size),  # Normalize concatenated features
            nn.Linear(hidden_size + input_size, output_size)  # Project to final output
        )
        # Optional post-processing normalization and activation
        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the ResChannelLSTM model.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, input_size), 
                        where T is the number of vertical levels.
        
        Returns:
            Tensor: Output of shape (B, T, output_size)
        """
        # Freeze certain input variables by detaching them from gradient computation
        # x_const1 = x[:, :, 0].unsqueeze(-1).detach()  # e.g., temperature (frozen)
        x_var1   = x[:, :, 0].unsqueeze(-1)           # e.g., humidity (trainable)
        x_var2   = x[:, :, 1].unsqueeze(-1)           # e.g., temperature (trainable)
        x_const3 = x[:, :, 2].unsqueeze(-1).detach()  # e.g., zenith angle (frozen)

        # Reconstruct input with selected trainable and frozen variables
        x = torch.cat([x_var1, x_var2, x_const3], dim=-1)  # (B, T, 3)

        skip = x  # Preserve original input for final residual connection

        # Project to hidden dimension
        x = self.linear(x)  # (B, T, hidden_size)

        # Pass through multiple BiLSTM blocks
        for layer in self.reslstm:
            x = layer(x)  # Each block applies BiLSTM + residual + norm

        # Optional final normalization and activation
        x = self.norm(x)
        x = self.relu(x)

        # Concatenate original input (skip connection) with processed features
        x = torch.cat([skip, x], dim=2)  # (B, T, 3 + hidden_size)

        # Final projection to output space
        return self.fc(x)  # (B, T, output_size)    

class MLPBlock(nn.Module):
    """
    A feed-forward MLP block that processes the entire sequence flattened across time steps.
    Designed for processing sequences of fixed length (e.g., 100 vertical levels).
    Applies two linear layers with ReLU and LayerNorm, and includes a residual connection.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the MLPBlock.

        This block reshapes the input (B, T, D) into (B, T*D) to apply fully connected layers
        across the entire sequence. It uses a fixed sequence length assumption (T=100).

        Args:
            input_size (int): Feature dimension per time step (e.g., 4).
            hidden_size (int): Hidden layer feature dimension.
            output_size (int): Output feature dimension per time step.
        """
        super(MLPBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # First FC: maps flattened input (B, 100 * input_size) -> (B, 100 * hidden_size)
        self.fc1 = nn.Linear(input_size * 100, hidden_size * 100)
        
        # Second FC: includes residual connection (input_size included in dim)
        # Input dim: (hidden_size + input_size) * 100
        self.fc2 = nn.Linear((hidden_size + input_size) * 100, output_size * 100)
        
        # Layer normalization after first linear layer
        self.norm = nn.LayerNorm(hidden_size * 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass with global residual connection.

        Args:
            x (Tensor): Input tensor of shape (B, T, input_size), where T=100.

        Returns:
            Tensor: Output of shape (B, T, output_size)
        """
        B, T, _ = x.shape  # T should be 100
        assert T == 100, f"Expected sequence length 100, got {T}"

        # Flatten temporal and feature dimensions: (B, T, input_size) -> (B, 100 * input_size)
        x = x.reshape(-1, self.input_size * 100)
        skip = x  # Global residual: store original flattened input

        # First transformation: linear -> norm -> activation
        x = self.fc1(x)           # (B, 100 * hidden_size)
        x = self.norm(x)          # Normalize
        x = self.relu(x)          # Activate

        # Concatenate residual before final projection
        x = torch.cat([skip, x], dim=1)  # (B, 100*(input_size + hidden_size))

        # Final linear projection
        x = self.fc2(x)  # (B, 100 * output_size)

        # Reshape back to sequence format
        return x.reshape(B, 100, self.output_size)  # (B, T, output_size)


class ResChannelMLP(nn.Module):
    """
    Residual Channel MLP model for processing atmospheric profiles.
    Uses frozen variables and deep MLP blocks with global residual connections.
    Designed for fixed-length sequences (T=100).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the ResChannelMLP model.

        Args:
            input_size (int): Number of input features (after variable selection/freeze).
            hidden_size (int): Dimension of hidden representations.
            num_layers (int): Number of stacked MLPBlocks.
            output_size (int): Number of output features.
        """
        super(ResChannelMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Linear projection from input space to hidden space
        self.linear = nn.Linear(input_size, hidden_size)
        
        # Stack of MLP blocks (each operates on flattened sequence)
        self.mlp_blocks = nn.ModuleList([
            MLPBlock(hidden_size, hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Final output layer with LayerNorm and linear projection
        # Includes residual connection from original input (hence input_size in dim)
        self.fc = nn.Sequential(
            nn.LayerNorm((hidden_size + input_size) * 100),
            nn.Linear((hidden_size + input_size) * 100, output_size * 100)
        )
        
        # Optional post-processing normalization and activation
        self.norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the ResChannelMLP model.

        Args:
            x (Tensor): Input tensor of shape (B, T, input_size), where T=100.

        Returns:
            Tensor: Output of shape (B, T, output_size)
        """
        B, T, _ = x.shape
        assert T == 100, f"Expected 100 vertical levels, got {T}"

        # === Variable Selection & Freezing ===
        # Keep some variables trainable, freeze others by detaching gradients
        x_var1   = x[:, :, 0].unsqueeze(-1)           # e.g., humidity (trainable)
        x_var2   = x[:, :, 1].unsqueeze(-1)           # e.g., temperature (trainable)
        x_const3 = x[:, :, 2].unsqueeze(-1).detach()  # e.g., zenith angle (frozen)

        # Reconstruct input with selected variables
        x = torch.cat([x_var1, x_var2, x_const3], dim=-1)  # (B, T, 3)
        skip = x  # Store original input for final residual connection

        # Project to hidden dimension
        x = self.linear(x)  # (B, T, hidden_size)

        # Pass through multiple MLP blocks (each flattens and processes globally)
        for layer in self.mlp_blocks:
            x = layer(x)  # Each block expects (B, 100, D), outputs same

        # Optional final normalization and activation
        x = self.norm(x)
        x = self.relu(x)

        # Concatenate original input (residual) with processed features
        x = torch.cat([skip, x], dim=2)  # (B, T, 3 + hidden_size)

        # Flatten for final FC layer
        x = x.reshape(-1, (self.hidden_size + self.input_size) * 100)

        # Final projection
        x = self.fc(x)  # (B * 100, output_size * 100) -> actually (B, output_size*100)

        # Reshape to batch and sequence format
        return x.reshape(B, 100, self.output_size)  # (B, T, output_size)

def compute_jacobian_fast(model, inputs ,is_create_graph=True):
    """
    正确的高效雅可比计算方法
    
    参数:
        model: 接受 (batch_size, seq_len, input_size) 的模型
        inputs: (batch_size, seq_len, input_size)
    
    返回:
        jacobian: (batch_size, seq_len, output_size, input_size)
    """
    # 确保输入需要梯度
    inputs = inputs.clone().requires_grad_(True)
    inputs.retain_grad()
    # 前向传播
    with torch.backends.cudnn.flags(enabled=False):
        outputs = model(inputs)  # (B, T, D)
    B, T, D = outputs.shape
    I = inputs.shape[-1]
    
    # 初始化雅可比矩阵
    jacobian = torch.zeros(B, T, D, I, device=inputs.device)
    
    # 为每个输出维度计算梯度
    for d in range(D):
        # 清零梯度
        if inputs.grad is not None:
            inputs.grad.zero_()
        
        # 计算单个输出维度对所有输入的梯度
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[:, :, d] = 1.0  # 只保留当前输出维度的梯度
        
        # 反向传播
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=is_create_graph,
            allow_unused=False
        )[0]  # (B, T, I)
        
        # 存储结果
        jacobian[:, :, d, :] = gradients
    
    return jacobian

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.000001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss improved: {self.val_loss_min:.6f} -> {val_loss:.6f}')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_model(device,X_train, Y_train, X_val, Y_val, model, batch_size=32, num_epochs=200):
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    
    # 初始化模型
    model = model.to(device)
    # 损失函数和优化器
    criterion = nn.MSELoss()#hybrid_loss  # 也可以使用 hybrid_loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)#torch.optim.Adadelta(model.parameters(), lr=1, rho=0.9, eps=1e-06, weight_decay=0)#
    
    # 学习率调度器和早停
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)
        
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        rel_errors = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                epoch_val_loss += loss.item() * x_val.size(0)
                
                # 计算相对误差
                rel_error = (outputs - y_val).abs() / (y_val.abs() + 1e-6)
                rel_errors.extend(rel_error.cpu().numpy().flatten())
        
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        # 检查是否是最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save(model.state_dict(), f'./Reslstm_TotOD_V4_ScanAngle.pth')
            print(f"Saved new best model with val loss: {best_val_loss:.6f}")
        
        # 早停检查
        early_stopping(avg_val_loss, model, f'./DL_MODEL/Best_Model_EarlyStop.pth')
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # 打印训练进度
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"Best Val Loss: {best_val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return model, train_losses, val_losses

def reverse_jacobian(
    jacobian: torch.Tensor,
    x_scale: torch.Tensor,
    y_scale: torch.Tensor
) -> torch.Tensor:
    """
    Reverse standardization of Jacobian from normalized to physical space.

    Args:
        jacobian: Jacobian in normalized space, shape (B, L, C, P)
        x_scale: Input scaler scales, shape (L, P)
        y_scale: Output scaler scales, shape (L, C)

    Returns:
        Jacobian in physical space.
    """
    batch_size, seq_len, channels, input_dim = jacobian.shape

    # Expand for broadcasting: (1, L, C, 1) and (1, L, 1, P)
    y_scale = y_scale.unsqueeze(0).unsqueeze(-1)  # (1, L, C, 1)
    x_scale = x_scale.unsqueeze(0).unsqueeze(-2)  # (1, L, 1, P)

    # dY/dX_physical = (σ_Y / σ_X) * dY_hat/dX_hat
    jacobian_phys = jacobian * y_scale / x_scale
    return jacobian_phys

def Jacobian_regularizer(
    model: nn.Module,
    inputs: torch.Tensor,
    jacobian_ref: torch.Tensor,
    scaler_x: torch.Tensor,
    scaler_y: torch.Tensor,
    OD_MIN,
    OD_MAX,
    target_dims: Optional[list] = None,
    is_create_graph = True,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    H¹ Regularization: Enforce physical consistency by matching model Jacobian
    against reference (ARMS) Jacobian in physical space.

    Args:
        model: Trained neural network
        inputs: Input tensor (B, L, P)
        jacobian_ref: Reference Jacobian (B, L, C, P)
        scaler_x: Input scale tensor (L, P)
        scaler_y: Output scale tensor (L, C)
        target_dims: Optional list of input dimensions to regularize
        epsilon: Small value to avoid division by zero

    Returns:
        Scalar loss (MAE between normalized Jacobians)
    """
    # 1. Compute current model Jacobian
    jacobian = compute_jacobian_fast(model, inputs, is_create_graph= is_create_graph)  # (B, L, C, P)

    # 2. Transform both Jacobians to physical space
    jacobian_phys = reverse_jacobian(jacobian, scaler_x, scaler_y)
    jacobian_ref_phys = reverse_jacobian(jacobian_ref, scaler_x, scaler_y)

   # 3. 初始化归一化后的 Jacobian
    jacobian_norm = torch.zeros_like(jacobian_phys)
    jacobian_ref_norm = torch.zeros_like(jacobian_ref_phys)

    batch_size, seq_len, output_size, input_size = jacobian_phys.shape

    # 4. 按通道归一化（output_channel, input_dim）
    for o in range(output_size):
        for i in range(input_size):
            # 提取当前通道的 Jacobian
            jac_pred = jacobian_phys[:, :, o, i]  # shape: (B, T)
            jac_ref = jacobian_ref_phys[:, :, o, i]

            # 计算 min 和 max（统一使用参考值）
            min_val = OD_MIN[o,i]
            max_val = OD_MAX[o,i]
            range_val = max_val - min_val

            # 如果 range_val 为 0（即全零通道），则使用 epsilon 代替
            if range_val.abs() < epsilon:
                jacobian_norm[:, :, o, i] = torch.log(torch.abs(jac_pred)+1) / (1.0 + epsilon)
                jacobian_ref_norm[:, :, o, i] = torch.log(torch.abs(jac_ref)+1) / (1.0 + epsilon)
            else:
                jacobian_norm[:, :, o, i] = (jac_pred - min_val) / range_val
                jacobian_ref_norm[:, :, o, i] = (jac_ref- min_val) / range_val

    # 5. 选择特定输入维度
    if target_dims is not None:
        jacobian_norm = jacobian_norm[:, :, :, target_dims]
        jacobian_ref_norm = jacobian_ref_norm[:, :, :, target_dims]

    # 6. 计算物理一致性损失（MAE）
    loss = torch.abs(jacobian_norm - jacobian_ref_norm).mean()

    return loss

def convert_JOD_to_Jnorm(Q_TL,T_TL,n_ang,scalerX_std,scalerY_std):

    n_profile, n_lay, n_channel = Q_TL.shape

    scale_Q_TL = np.zeros((n_profile, n_lay, n_channel))
    scale_T_TL = np.zeros((n_profile, n_lay, n_channel))

    for nl in range(n_lay):
        for nc in range(n_channel):
            scale_Q_TL[:, nl, nc] = scalerX_std[nl, 0] * Q_TL[:, nl, nc] / scalerY_std[nl, nc]
            scale_T_TL[:, nl, nc] = scalerX_std[nl, 1] * T_TL[:, nl, nc] / scalerY_std[nl, nc]

    # Expand across angular dimension
    jacobian_Q = np.repeat(scale_Q_TL[:, np.newaxis, :, :], n_ang, axis=1)  # (N, Ang, L, C)
    jacobian_T = np.repeat(scale_T_TL[:, np.newaxis, :, :], n_ang, axis=1)

    # Reshape to (N * Ang, L, C)
    jacobian_Q = jacobian_Q.reshape(-1, n_lay, n_channel)
    jacobian_T = jacobian_T.reshape(-1, n_lay, n_channel)

    # Stack into full Jacobian: (B, L, C, P)
    Jacobian = np.zeros((n_profile * n_ang, n_lay, n_channel,3))
    Jacobian[:,:,:,(0,1)] = np.stack([jacobian_Q, jacobian_T], axis=-1)  # (B, L, C, 2)
    J_tensor = torch.tensor(Jacobian, dtype=torch.float32)
    return J_tensor

def train_model_Jacobian(device,X_train, Y_train,J_train, X_val, Y_val,J_val,J_MIN,
    J_MAX,model,scalerX_std,scalerY_std, batch_size=256, num_epochs=80,λ=0.005):
    
    # 创建带噪声注入的数据集
    # train_dataset = CustomDataset(X_train, Y_train, 
    #                              transform=AddGaussianNoise(std=0.01))
    train_dataset = TensorDataset(X_train, Y_train,J_train)
    val_dataset = TensorDataset(X_val, Y_val,J_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    
    # 初始化模型
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()#hybrid_loss  # 也可以使用 hybrid_loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, amsgrad=True)#torch.optim.Adadelta(model.parameters(), lr=1, rho=0.9, eps=1e-06, weight_decay=0)#
    
    # 学习率调度器和早停
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=25, verbose=True)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        epoch_J_loss = 0
        epoch_task_loss = 0
        for x_batch, y_batch, J_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            x_batch, y_batch, J_batch = x_batch.to(device), y_batch.to(device), J_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            J_loss = Jacobian_regularizer(model,x_batch, J_batch,scalerX_std.clone().detach().to(device),scalerY_std.clone().detach().to(device),J_MIN,J_MAX,target_dims=[0,1]) 
            task_loss =criterion(outputs, y_batch)
            
            loss =  task_loss + λ * J_loss
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)
            epoch_task_loss += task_loss.item() * x_batch.size(0)
            epoch_J_loss += J_loss.item() * x_batch.size(0)
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_task_loss = epoch_task_loss / len(train_loader.dataset)
        avg_J_loss = epoch_J_loss / len(train_loader.dataset)
        # λ = 0.01*avg_J_loss / avg_task_loss  # 初始 λ
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        rel_errors = []

        for x_val, y_val, J_val in val_loader:
            x_val, y_val, J_val = x_val.to(device), y_val.to(device), J_val.to(device)
            with torch.no_grad():
                outputs = model(x_val)
                task_loss = criterion(outputs, y_val)
            
            J_loss = Jacobian_regularizer(model.to(device),x_val.clone().detach().requires_grad_(True).to(device), J_val,scalerX_std.clone().detach().to(device),scalerY_std.clone().detach().to(device),J_MIN,J_MAX,target_dims=[0,1],is_create_graph=False) 
            loss =  task_loss + λ * J_loss

            epoch_val_loss += loss.item() * x_val.size(0)
        
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        # 检查是否是最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'./DL_MODEL/Best_Model_Jacobian_EarlyStop.pth')
            print(f"Saved new best model with val loss: {best_val_loss:.6f}")
        
        # 早停检查
        # early_stopping(avg_val_loss, model, f'./WorkPath/Model/ResMLP_TotOD_V3_hidden_size{hidden_size}_num_layers{num_layers}_EarlyStop_Jacobian.pth')
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break
        
        # 打印训练进度
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Jacobian Loss: {avg_J_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"Best Val Loss: {best_val_loss:.6f}, "
              f"λ: {λ:.3f}"
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return model, train_losses, val_losses


