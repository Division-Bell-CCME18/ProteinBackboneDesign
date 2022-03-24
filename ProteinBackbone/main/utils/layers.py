import os
from typing import Callable, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_scatter import scatter_mean, scatter_add


# common
class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).

        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).

        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output



class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.

    Note there is no activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x



# gin
class GINEConv(MessagePassing):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 activation="softplus", **kwargs):
        super(GINEConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None       

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)        


class GraphIsomorphismNetwork(torch.nn.Module):

    def __init__(self, hidden_dim, num_convs=3, activation="softplus", readout="sum", short_cut=False, concat_hidden=False):
        super(GraphIsomorphismNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        


        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(GINEConv(MultiLayerPerceptron(hidden_dim, [hidden_dim, hidden_dim], \
                                    activation=activation), activation=activation))

        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    

    def forward(self, data, node_attr, edge_attr):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """
 
        hiddens = []
        conv_input = node_attr # (num_node, hidden)

        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, data.edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape                
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden += conv_input

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        graph_feature = self.readout(data, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


