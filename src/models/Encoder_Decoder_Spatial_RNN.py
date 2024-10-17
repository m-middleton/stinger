import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PerChannelEncoder(nn.Module):
    def __init__(self, nirs_input_dim, hidden_size, num_layers=1):
        super(PerChannelEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=nirs_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, nirs_inputs):
        # nirs_inputs shape: (batch_size, nirs_seq_len, nirs_channels)
        batch_size, nirs_seq_len, nirs_channels = nirs_inputs.size()
        
        # Reshape to process all channels separately
        nirs_inputs = nirs_inputs.permute(2, 0, 1)  # (nirs_channels, batch_size, nirs_seq_len)
        nirs_inputs = nirs_inputs.contiguous().view(-1, nirs_seq_len, 1)  # (nirs_channels * batch_size, nirs_seq_len, nirs_input_dim_per_channel)
        
        outputs, (hidden, cell) = self.lstm(nirs_inputs)
        
        # Reshape back to original dimensions
        outputs = outputs.view(nirs_channels, batch_size, nirs_seq_len, self.hidden_size)
        outputs = outputs.permute(1, 0, 2, 3)  # (batch_size, nirs_channels, nirs_seq_len, hidden_size)
        
        return outputs  # Shape: (batch_size, nirs_channels, nirs_seq_len, hidden_size)

class SpatialAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, nirs_channels, nirs_seq_len, eeg_channels):
        super(SpatialAttention, self).__init__()
        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)
        self.lambda_param = nn.Parameter(torch.tensor(1.0))
        self.nirs_channels = nirs_channels
        self.nirs_seq_len = nirs_seq_len
        self.eeg_channels = eeg_channels

    def forward(self, hidden, encoder_outputs, distance_matrix_expanded):
        # hidden shape: (batch_size, decoder_hidden_size)
        # encoder_outputs shape: (batch_size, nirs_channels * nirs_seq_len, encoder_hidden_size)
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)  # nirs_channels * nirs_seq_len

        # Repeat hidden state
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, decoder_hidden_size)

        # Compute energy scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, decoder_hidden_size)
        energy = self.v(energy).squeeze(2)  # (batch_size, seq_len)

        # Expand energy for EEG channels
        energy = energy.unsqueeze(2).repeat(1, 1, self.eeg_channels)  # (batch_size, seq_len, eeg_channels)

        # Expand distance_matrix for batch
        distance_matrix_batch = distance_matrix_expanded.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, seq_len, eeg_channels)

        # Adjust energy with spatial distances
        adjusted_energy = energy - self.lambda_param * distance_matrix_batch

        # Compute attention weights
        attn_weights = F.softmax(adjusted_energy, dim=1)  # (batch_size, seq_len, eeg_channels)

        return attn_weights

class Decoder(nn.Module):
    def __init__(self, eeg_output_dim, encoder_hidden_size, decoder_hidden_size, nirs_channels, nirs_seq_len, eeg_channels, num_layers=1):
        super(Decoder, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.eeg_channels = eeg_channels
        self.eeg_output_dim = eeg_output_dim  # Features per EEG channel
        self.attention = SpatialAttention(encoder_hidden_size, decoder_hidden_size, nirs_channels, nirs_seq_len, eeg_channels)
        
        self.lstm = nn.LSTM(
            input_size=eeg_output_dim * eeg_channels,  # Total features for all EEG channels
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size, eeg_output_dim * eeg_channels)
        self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size, eeg_output_dim)

        
    def forward(self, input, hidden, cell, encoder_outputs, distance_matrix_expanded):
        # input shape: (batch_size, 1, eeg_output_dim * eeg_channels)
        # hidden shape: (num_layers, batch_size, decoder_hidden_size)
        
        # Compute attention weights
        attn_weights = self.attention(hidden[-1], encoder_outputs, distance_matrix_expanded)
        # attn_weights shape: (batch_size, seq_len, eeg_channels)
        
        # Compute context vectors
        context = torch.einsum('bse,bsh->beh', attn_weights, encoder_outputs)  # (batch_size, eeg_channels, encoder_hidden_size)
        
        # LSTM output
        output, (hidden, cell) = self.lstm(input, (hidden, cell))  # output: (batch_size, 1, decoder_hidden_size)
        output = output.squeeze(1)  # (batch_size, decoder_hidden_size)
        
        # Expand output to match EEG channels
        output_expanded = output.unsqueeze(1).repeat(1, self.eeg_channels, 1)  # (batch_size, eeg_channels, decoder_hidden_size)

        # Concatenate context and output
        combined = torch.cat((output_expanded, context), dim=2)  # (batch_size, eeg_channels, decoder_hidden_size + encoder_hidden_size)

        # Generate prediction
        prediction = self.fc_out(combined)  # (batch_size, eeg_channels, eeg_output_dim)

        # reshape to (batch_size, eeg_output_dim, eeg_channels)
        if self.eeg_output_dim == 1:
            prediction = prediction.squeeze(2)
        else:
            prediction = prediction.reshape(-1, self.eeg_output_dim, self.eeg_channels)

        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, nirs_inputs, eeg_targets, distance_matrix_expanded, teacher_forcing_ratio=0.5):
        batch_size = nirs_inputs.size(0)
        eeg_seq_len = eeg_targets.size(1)
        eeg_channels = eeg_targets.size(2)
        
        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, eeg_seq_len, eeg_channels).to(self.device)
        
        # Encoder forward pass
        encoder_outputs = self.encoder(nirs_inputs)
        # Flatten encoder outputs
        batch_size, nirs_channels, nirs_seq_len, hidden_size = encoder_outputs.size()
        # encoder_outputs = encoder_outputs.view(batch_size, nirs_channels * nirs_seq_len, hidden_size)        
        encoder_outputs = encoder_outputs.reshape(batch_size, nirs_channels * nirs_seq_len, hidden_size)
        
        # Initial decoder input
        input = eeg_targets[:, 0, :].unsqueeze(1)  # (batch_size, 1, eeg_channels)
        hidden = torch.zeros(self.decoder.num_layers, batch_size, self.decoder.decoder_hidden_size).to(self.device)
        cell = torch.zeros(self.decoder.num_layers, batch_size, self.decoder.decoder_hidden_size).to(self.device)
        
        for t in range(1, eeg_seq_len):
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs, distance_matrix_expanded)
            outputs[:, t, :] = output  # (batch_size, eeg_channels)
            
            # Teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            input = eeg_targets[:, t, :].unsqueeze(1) if teacher_force else output.unsqueeze(1)
        
        return outputs


def inference(model, data_loader, device, distance_matrix_expanded):
    """
    Performs inference using the trained model to predict EEG data from NIRS data.

    Parameters:
    - model: The trained Seq2Seq model.
    - data_loader: DataLoader providing NIRS inputs and EEG targets with batch size of 1.
                   Each batch should be a dictionary with keys 'nirs_inputs' and 'eeg_targets'.
    - device: The device to run the inference on ('cuda' or 'cpu').
    - distance_matrix_expanded: The expanded distance matrix for spatial attention.

    Returns:
    - targets_array: NumPy array of shape (samples, sequence_length, channels) containing the target EEG data.
    - predictions_array: NumPy array of shape (samples, sequence_length, channels) containing the predicted EEG data.
    """

    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (nirs_inputs, eeg_targets) in enumerate(data_loader):
            # Move data to the specified device
            # nirs_inputsShape: (1, nirs_seq_len, nirs_input_dim)
            # eeg_targetsShape: (1, eeg_seq_len, eeg_channels * eeg_output_dim)

            # Forward pass with teacher forcing turned off (use model's own predictions)
            outputs = model(nirs_inputs, eeg_targets, distance_matrix_expanded, teacher_forcing_ratio=0.0)
            # outputs shape: (1, eeg_seq_len, eeg_channels * eeg_output_dim)

            # Reshape outputs and targets if necessary
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)  # Ensure outputs are of shape (batch_size, seq_len, eeg_channels * eeg_output_dim)
            eeg_targets = eeg_targets.view(eeg_targets.size(0), eeg_targets.size(1), -1)

            # Move outputs and targets to CPU and convert to NumPy arrays
            predictions = outputs.squeeze(0).cpu().numpy()  # Shape: (eeg_seq_len, eeg_channels * eeg_output_dim)
            targets = eeg_targets.squeeze(0).cpu().numpy()  # Shape: (eeg_seq_len, eeg_channels * eeg_output_dim)

            # If eeg_output_dim > 1, reshape to separate channels and features per channel
            if model.decoder.eeg_output_dim > 1:
                predictions = predictions.reshape(predictions.shape[0], model.decoder.eeg_channels, model.decoder.eeg_output_dim)
                targets = targets.reshape(targets.shape[0], model.decoder.eeg_channels, model.decoder.eeg_output_dim)
            else:
                # Reshape to (sequence_length, channels)
                predictions = predictions.reshape(predictions.shape[0], model.decoder.eeg_channels)
                targets = targets.reshape(targets.shape[0], model.decoder.eeg_channels)

            # Append to the lists
            all_predictions.append(predictions)  # Each element is (sequence_length, channels)
            all_targets.append(targets)          # Each element is (sequence_length, channels)

    # Convert lists to NumPy arrays
    predictions_array = np.array(all_predictions)  # Shape: (samples, sequence_length, channels)
    targets_array = np.array(all_targets)          # Shape: (samples, sequence_length, channels)

    return targets_array, predictions_array