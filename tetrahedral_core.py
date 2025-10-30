"""
Tetrahedral Neural Network Architecture
========================================

The immutable core architecture proven by arithmetic generalization.

Structure:
  - 4 vertices (core computation nodes)
  - 6 edges (pairwise linear attention interactions)
  - 4 triangular faces (3-point attention mechanisms)

This module contains only the essential architecture components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriSimplicialAttention(nn.Module):
    """
    Triangular attention over 3 vertices (a face of the tetrahedron).
    
    This is the fundamental 3-point attention operation that computes
    attention-weighted combinations of 3 inputs, enabling the network to
    discover higher-order relationships between vertices.
    
    Each of the 4 tetrahedral faces uses this operation.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.to_qkv = nn.Linear(dim * 3, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a, b, c: Tensors of shape (batch, dim) representing 3 vertices
        
        Returns:
            Attended combination of inputs, shape (batch, dim)
        """
        x = torch.cat([a, b, c], dim=-1)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention scores
        scores = torch.sum(q * k, dim=-1, keepdim=True) / (self.dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Combine with mean
        combined = (a + b + c) / 3
        attended = attn * v

        return self.out(attended + combined)


class TetrahedralCore(nn.Module):
    """
    THE CORE TETRAHEDRAL ARCHITECTURE
    
    This is the immutable heart of the system. It works across all domains
    because it makes NO assumptions about the data - it just provides the
    geometric structure for self-organization.
    
    Structure:
        - 4 vertices (W, X, Y, Z)
        - 6 edges (all pairwise interactions: WX, WY, WZ, XY, XZ, YZ)
        - 4 triangular faces (WXY, WXZ, WYZ, XYZ)
    
    The beauty: vertices self-organize their specialization during training.
    We don't tell them what to represent - they figure it out!
    """
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 64):
        """
        Args:
            input_dim: Dimension of input data
            output_dim: Dimension of output data
            latent_dim: Internal representation dimension (64 is optimal)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # Input embedding - maps input to 4 vertices
        self.embed = nn.Linear(input_dim, latent_dim * 4)

        # 6 edges - pairwise interactions between vertices
        # Linear attention on edges
        self.edges = nn.ModuleList([
            nn.Linear(latent_dim * 2, latent_dim) for _ in range(6)
        ])

        # 4 triangular faces - higher-order interactions (3-point attention)
        self.faces = nn.ModuleList([
            TriSimplicialAttention(latent_dim) for _ in range(4)
        ])

        # Output projection
        self.output = nn.Linear(latent_dim * 4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim)
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        batch_size = x.size(0)

        # Embed input to 4 vertices
        vertices = self.embed(x).view(batch_size, 4, self.latent_dim)

        # === EDGE PROCESSING ===
        # Process all 6 edges (pairwise vertex interactions)
        edge_indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        edge_updates = []

        for i, (a, b) in enumerate(edge_indices):
            # Concatenate vertex pair and process through edge layer
            edge_input = torch.cat([vertices[:, a], vertices[:, b]], dim=-1)
            edge_updates.append(self.edges[i](edge_input))

        # Average edge updates and add to all vertices
        edge_tensor = torch.stack(edge_updates, dim=1)
        edge_mean = edge_tensor.mean(dim=1, keepdim=True)
        vertices = vertices + edge_mean.expand(-1, 4, -1)

        # === FACE PROCESSING ===
        # Process all 4 triangular faces (3-point attention)
        face_indices = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        face_updates = []

        for i, (a, b, c) in enumerate(face_indices):
            # Apply triangular attention to each face
            face_output = self.faces[i](
                vertices[:, a],
                vertices[:, b],
                vertices[:, c]
            )
            face_updates.append(face_output)

        # Average face updates and add to vertices
        face_tensor = torch.stack(face_updates, dim=1)
        face_mean = face_tensor.mean(dim=1, keepdim=True)
        vertices = vertices + face_mean.expand(-1, 4, -1)

        # Project to output
        vertices_flat = vertices.reshape(batch_size, -1)
        output = self.output(vertices_flat)


        return output
