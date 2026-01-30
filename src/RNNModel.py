import torch
import torch.nn as nn
import torch.nn.functional as F

class LogicCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LogicCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Logic gates
        self.and_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.or_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.not_gate = nn.Linear(hidden_size, hidden_size)
        
        # LSTM components
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden, cell):
        combined = torch.cat((x, hidden), dim=1)
        
        # Logic operations
        and_out = torch.sigmoid(self.and_gate(combined))
        or_out = torch.sigmoid(self.or_gate(combined))
        not_out = torch.tanh(self.not_gate(hidden))
        
        # LSTM operations with logic integration
        forget = torch.sigmoid(self.forget_gate(combined))
        input_g = torch.sigmoid(self.input_gate(combined))
        cell_tilde = torch.tanh(self.cell_gate(combined))
        
        # Combine logic and LSTM
        cell_new = forget * cell + input_g * cell_tilde
        cell_logic = and_out * cell_new + or_out * not_out
        
        output = torch.sigmoid(self.output_gate(combined))
        hidden_new = output * torch.tanh(cell_logic)
        
        return hidden_new, cell_logic

class LogicRNNLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_layers=2, dropout=0.5):
        """
        Logic RNN+LSTM model for deepfake detection
        Args:
            input_size (int): Size of input features (default: 1024 for typical face embeddings)
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout rate between layers
        """
        super(LogicRNNLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Logic LSTM layers
        self.logic_cells = nn.ModuleList([
            LogicCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x, lengths=None):
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            lengths (torch.Tensor): Lengths of each sequence in the batch
        Returns:
            torch.Tensor: Model predictions
        """
        batch_size, seq_length, _ = x.size()
        
        if lengths is not None:
            # Sort sequences by length for packed sequence
            lengths, sort_idx = lengths.sort(0, descending=True)
            x = x[sort_idx]
            
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Process sequence through Logic LSTM layers
        outputs = []
        for t in range(seq_length):
            h_temp = h
            c_temp = c
            x_t = x[:, t, :]
            
            # Pass through all layers
            for i, logic_cell in enumerate(self.logic_cells):
                h_temp, c_temp = logic_cell(x_t if i == 0 else h_temp, h_temp, c_temp)
                if i < self.num_layers - 1:  # Apply dropout between layers
                    h_temp = self.dropout(h_temp)
            
            outputs.append(h_temp)
            h, c = h_temp, c_temp
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        if lengths is not None:
            # Create a mask for valid timesteps
            mask = torch.arange(seq_length).expand(batch_size, seq_length).to(x.device)
            mask = mask < lengths.unsqueeze(1)
            mask = mask.float().unsqueeze(-1)
            outputs = outputs * mask
        
        # Apply attention
        attention_weights = self.attention(outputs)
        context = torch.sum(attention_weights * outputs, dim=1)
        
        # Final classification
        output = self.classifier(context)
        return torch.sigmoid(output)

    def predict(self, x, lengths=None):
        """
        Make binary predictions
        Args:
            x (torch.Tensor): Input tensor
            lengths (torch.Tensor): Optional sequence lengths
        Returns:
            torch.Tensor: Binary predictions (0 or 1)
        """
        with torch.no_grad():
            outputs = self.forward(x, lengths)
            predictions = (outputs >= 0.5).float()
        return predictions

def create_model(config=None):
    """
    Factory function to create an instance of LogicRNNLSTM
    Args:
        config (dict): Configuration dictionary with model parameters
    Returns:
        LogicRNNLSTM: Instance of the model
    """
    if config is None:
        config = {
            'input_size': 1024,
            'hidden_size': 512,
            'num_layers': 2,
            'dropout': 0.5
        }
    
    return LogicRNNLSTM(
        input_size=config.get('input_size', 1024),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.5)
    )