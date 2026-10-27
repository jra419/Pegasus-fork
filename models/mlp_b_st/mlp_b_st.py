import torch
import torch.nn.functional as F
from torch import nn

TREE_DEPTH = 4
NUM_SPLITS = 2 ** TREE_DEPTH - 1
NUM_LEAVES = 2 ** TREE_DEPTH

# A training split answers "yes" only when its value exceeds the threshold by more than this;
# mlp_b_st_lut_convert adds it to the thresholds, so LutLayer's plain "> T" gives the same answers.
SPLIT_MARGIN = 0.0001

HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16


# Forward pass uses hard, backward pass uses the gradient of soft.
def straight_through(hard, soft):
    return (hard - soft).detach() + soft

# Training-time quantized layer: picks a leaf prototype per codeblock, with
# straight-through estimators for the tree splits and the leaf choice.
class TreeQuantizedLayer(nn.Module):
    def __init__(self, num_codeblocks, codeblock_dim, device):
        super().__init__()

        self.register_buffer('S', torch.randn(num_codeblocks, codeblock_dim, NUM_SPLITS).to(device))
        self.register_buffer('H', torch.randn(NUM_SPLITS, NUM_LEAVES).to(device))

        self.num_codeblocks = num_codeblocks
        self.codeblock_dim  = codeblock_dim
        self.T              = nn.Parameter(torch.randn(num_codeblocks, NUM_SPLITS).to(device))
        self.LUT            = nn.Parameter(torch.randn(num_codeblocks, NUM_LEAVES,
                                                       codeblock_dim).to(device))

    # b=batch, c=codeblock, d=codeblock dimension, s=split node, k=leaf.
    def forward(self, x):
        x           = x.view(-1, self.num_codeblocks, self.codeblock_dim)
        x           = torch.einsum('bcd,cds->bcs', x, self.S)
        x           = x - self.T - SPLIT_MARGIN
        x           = straight_through(torch.sign(x), torch.tanh(x))
        x           = torch.einsum('bcs,sk->bck', x, self.H)
        one_hot     = F.one_hot(torch.argmax(x, dim=-1), num_classes=NUM_LEAVES).float()
        softmax     = torch.softmax(x, dim=-1)
        x_one_hot   = torch.einsum('bck,ckd->bcd', one_hot, self.LUT)
        x_softmax   = torch.einsum('bck,ckd->bcd', softmax, self.LUT)
        x           = straight_through(x_one_hot, x_softmax)
        x           = x.reshape(-1, self.num_codeblocks * self.codeblock_dim)

        return x

# Staged training: stage N has the first N-1 linear layers tree-quantized. Every stage but
# the last also returns the activations the next stage's trees are fitted on.

# Stage 1: plain float MLP.
class MlpStage1(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device):
        super().__init__()

        self.device = device
        self.fc1    = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2    = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        self.relu   = nn.ReLU()
        self.norm0  = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1  = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2  = nn.BatchNorm1d(hidden_size2).to(self.device)

    def forward(self, x):
        x               = x.to(self.device)
        x               = self.norm0(x)
        quantizer_input = [x]
        x               = self.fc1(x)
        x               = self.norm1(x)
        x               = self.relu(x)
        x               = self.fc2(x)
        x               = self.norm2(x)
        x               = self.relu(x)
        x               = self.output(x)

        return F.log_softmax(x, dim=-1), quantizer_input

# Stage 2: fc1's input is tree-quantized.
class MlpStage2(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device, dim1):
        super().__init__()

        self.device = device
        self.fc1    = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2    = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        self.relu   = nn.ReLU()
        self.norm0  = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1  = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2  = nn.BatchNorm1d(hidden_size2).to(self.device)

        self.MM1 = TreeQuantizedLayer(input_size // dim1, dim1, device)

    def forward(self, x):
        x               = x.to(self.device)
        x               = self.norm0(x)
        x               = self.MM1(x)
        x               = self.fc1(x)
        x               = self.norm1(x)
        x               = self.relu(x)
        quantizer_input = [x]
        x               = self.fc2(x)
        x               = self.norm2(x)
        x               = self.relu(x)
        x               = self.output(x)

        return F.log_softmax(x, dim=-1), quantizer_input

# Stage 3: fc1's and fc2's inputs are tree-quantized.
class MlpStage3(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device, dim1, dim2):
        super().__init__()

        self.device = device
        self.fc1    = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2    = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        self.relu   = nn.ReLU()
        self.norm0  = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1  = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2  = nn.BatchNorm1d(hidden_size2).to(self.device)

        self.MM1 = TreeQuantizedLayer(input_size // dim1, dim1, device)
        self.MM2 = TreeQuantizedLayer(hidden_size // dim2, dim2, device)

    def forward(self, x):
        x               = x.to(self.device)
        x               = self.norm0(x)
        x               = self.MM1(x)
        x               = self.fc1(x)
        x               = self.norm1(x)
        x               = self.relu(x)
        x               = self.MM2(x)
        x               = self.fc2(x)
        x               = self.norm2(x)
        x               = self.relu(x)
        quantizer_input = [x]
        x               = self.output(x)

        return F.log_softmax(x, dim=-1), quantizer_input

# Stage 4: all three linear layers' inputs are tree-quantized.
class MlpStage4(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device,
                 dim1, dim2, dim3):
        super().__init__()

        self.device = device
        self.fc1    = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2    = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        self.relu   = nn.ReLU()
        self.norm0  = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1  = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2  = nn.BatchNorm1d(hidden_size2).to(self.device)

        self.MM1 = TreeQuantizedLayer(input_size // dim1, dim1, device)
        self.MM2 = TreeQuantizedLayer(hidden_size // dim2, dim2, device)
        self.MM3 = TreeQuantizedLayer(hidden_size2 // dim3, dim3, device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.norm0(x)
        x = self.MM1(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.MM2(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.MM3(x)
        x = self.output(x)

        return F.log_softmax(x, dim=-1), 0

# Inference-time layer: hard tree traversal per codeblock, then the sum of the chosen
# leaves' LUT rows. The LUT already has the following linear layer folded in.
class LutLayer(nn.Module):
    def __init__(self, num_codeblocks, codeblock_dim, output_size, device):
        super().__init__()

        self.register_buffer('S', torch.randn(num_codeblocks, codeblock_dim, NUM_SPLITS).to(device))
        self.register_buffer('H', torch.randn(NUM_SPLITS, NUM_LEAVES).to(device))

        self.num_codeblocks = num_codeblocks
        self.codeblock_dim  = codeblock_dim
        self.T              = nn.Parameter(torch.randn(num_codeblocks, NUM_SPLITS).to(device))
        self.LUT            = nn.Parameter(torch.randn(num_codeblocks, NUM_LEAVES,
                                                       output_size).to(device))

    # b=batch, c=codeblock, d=codeblock dimension, s=split node, k=leaf, o=output.
    def forward(self, x):
        x           = x.view(-1, self.num_codeblocks, self.codeblock_dim)
        x           = torch.einsum('bcd,cds->bcs', x, self.S)
        x           = x - self.T
        x[x == 0]   = -1
        x           = torch.sign(x)
        x           = torch.einsum('bcs,sk->bck', x, self.H)
        one_hot     = F.one_hot(torch.argmax(x, dim=-1), num_classes=NUM_LEAVES).float()
        x           = torch.einsum('bck,cko->bco', one_hot, self.LUT)
        x           = torch.sum(x, dim=1)

        return x

# Fully table-based MLP, loaded from the state dict produced by mlp_b_st_lut_convert.
class MlpLut(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device,
                 dim1, dim2, dim3):
        super().__init__()

        self.device = device

        self.MM1 = LutLayer(input_size // dim1, dim1, hidden_size, device)
        self.MM2 = LutLayer(hidden_size // dim2, dim2, hidden_size2, device)
        self.MM3 = LutLayer(hidden_size2 // dim3, dim3, num_classes, device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.MM1(x)
        x = self.MM2(x)
        x = self.MM3(x)

        return F.log_softmax(x, dim=-1)
