import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_npu

# Principally referenced from Mistral MOE implementation
class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    
    def forward(self, hidden_states):
        # Calculate routing logits
        router_logits = self.gate(hidden_states)  # Shape: (batch * seq_len, expert_number)
        
        # Calculate probabilities of experts after softmax
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # Select top-k expert outputs
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # Both shapes: (batch * seq_len, top_k)
        
        # Normalize expert weights
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)
        
        # Generate expert mask
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  # Shape: (batch * seq_len, top_k, expert_number)
        expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, batch * seq_len)
        
        return router_logits, router_weights, selected_experts, expert_mask


# Routing load balancing loss
def switch_load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
   
    # Calculate routing probabilities
    router_probs = torch.softmax(router_logits, dim=-1)  # [batch * seq_len, num_experts]
    
    # Get top-2 experts for each token
    _, selected_experts = torch.topk(router_probs, k=2, dim=-1) 
    
    # Create one-hot matrix representing selected experts
    mask = torch.nn.functional.one_hot(selected_experts, num_experts).float() 
    
    # Calculate expected load for each expert (ideally 1/num_experts)
    expected_load = torch.ones_like(router_probs) / num_experts
    
    # Calculate actual load (number of tokens handled by each expert / total tokens)
    # Calculate mean across the batch dimension
    actual_load = mask.mean(dim=0)
    
    # Calculate auxiliary loss
    # Penalizes discrepancy between actual load distribution and expected load
    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * num_experts
    
    # Calculate z_loss (optional)
    # Penalizes excessively large routing logits
    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001  # Adjustable hyperparameter
    
    # Total loss
    total_loss = aux_loss + z_loss * z_loss_weight
    
    return total_loss


class BasicExpert(nn.Module):
    # An expert can be a simple linear layer, 
    # an MLP layer, or a more complex gated MLP (e.g., using SwiGLU)
    def __init__(self, hidden_dim, mult=2):
        super().__init__()
     
        # Using SiLU gated activation, supports dimension scaling
        # Two parallel linear layers (input -> expanded dimension)
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * mult))
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim * mult))
        # Output projection (scale back to hidden_dim)
        self.fc3 = nn.Linear(int(hidden_dim * mult), hidden_dim)
    
    def forward(self, x):
        # return self.linear(x)
        
        # Main branch * SiLU(Gating branch)
        x = self.fc1(x) * F.silu(self.fc2(x))
        # Project back to output dimension
        return self.fc3(x)

class SparseMOE(nn.Module):
    # Sparse MOE model where each token passes through top-k experts
    # to obtain the corresponding hidden embeddings.
    def __init__(self, 
                hidden_dim=128, 
                expert_hidden_mult = 2,
                top_k=2, 
                expert_number=4,                     
                # shared_experts_number=2, # Shared experts not used in SparseMOE
                experts_model  = None,
                # share_experts  = None,
                
                ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.expert_hidden_mult = expert_hidden_mult
        self.expert_number = expert_number
        self.top_k = top_k

        self.experts_model = experts_model
        if experts_model is None:
            self.experts_model = BasicExpert

        self.experts = nn.ModuleList(
            [
                self.experts_model(self.hidden_dim, self.expert_hidden_mult) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        hidden_states = x.reshape(-1, hidden_dim) # [Batch * Length, Dimension] 

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # selected_experts_indices shape: (batch * seq_len, top_k)
        # expert_mask shape: (expert_number, top_k, batch * seq_len)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape: (top_k, batch * seq_len)
            idx, top_x = torch.where(expert_mask[expert_idx]) 
          
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim) # (selected_token_number, hidden_dim)

            # router_weights shape: (batch * seq_len, top_k)
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  # (selected_token_number, 1) - Broadcasting applies

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        

        # Reshape final_hidden_states back to original shape
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits # Shape: (batch * seq_len, expert_number)



class ShareMOE(nn.Module):
    def __init__(self, 
                    hidden_dim=128, 
                    expert_hidden_mult = 2,
                    top_k=2, 
                    expert_number=4,                     
                    shared_experts_number=2, # Number of shared experts
                    experts_model  = None,
                    share_experts_model  = None, # Shared model type

                    ):
        super().__init__()

       
        self.expert_number = expert_number
        self.shared_experts_number = shared_experts_number
        self.share_experts_model = share_experts_model
        if share_experts_model is None:
            self.share_experts_model = BasicExpert

        self.moe_model = SparseMOE( hidden_dim=hidden_dim, 
                                    expert_hidden_mult = expert_hidden_mult,
                                    top_k=top_k, 
                                    expert_number=expert_number,                     
                                    experts_model=experts_model,
                                    )
        self.shared_experts = nn.ModuleList(
            [
                self.share_experts_model(
                    hidden_dim, expert_hidden_mult
                ) for _ in range(shared_experts_number)
            ]
        )

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        # First pass through the Sparse MOE model
        sparse_moe_out, router_logits = self.moe_model(x)
        
        # Apply shared experts to each token
        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ] # Output shape of each expert: (batch, seq_len, hidden_dim)
        
        shared_experts_out = torch.stack(
            shared_experts_out, dim=0
        ).sum(dim=0)

        aux_loss = switch_load_balancing_loss(router_logits, self.expert_number)
        
        # Sum sparse_moe_out and shared_experts_out
        return sparse_moe_out + shared_experts_out, aux_loss


def test_share_moe():
    x = torch.rand(2, 4, 16)
    share_expert_moe = ShareMOE( hidden_dim=x.shape[-1],
                                expert_hidden_mult = 0.5,
                                top_k=2, 
                                expert_number=4,                     
                                shared_experts_number=2,
                                experts_model  = None,
                                share_experts_model  = None,
                                )
    out, aux_loss = share_expert_moe(x)
    print('Model Structure:', share_expert_moe)
    print('Output Shape:', out.shape)
    print('Auxiliary Loss:', aux_loss)   


def test_moe_training():
    # Create a simple dataset
    batch_size = 32
    seq_len = 16
    hidden_dim = 32
    num_batches = 100
    
    model = ShareMOE( hidden_dim=hidden_dim,
                                expert_hidden_mult = 2,
                                top_k=2, 
                                expert_number=4,                     
                                shared_experts_number=2,
                                experts_model  = None,
                                share_experts_model  = None,
                                )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for batch in range(num_batches):
        # Generate random input and target data
        x = torch.randn(batch_size, seq_len, hidden_dim)
        target = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward pass
        output, aux_loss = model(x)

        # Compute losses
        # MSE loss for prediction
        mse_loss = F.mse_loss(output, target)
        
        # Combined loss
        total_loss = mse_loss + 0.01 * aux_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            print(f"Batch {batch}, Loss: {total_loss.item():.4f} "
                  f"(MSE: {mse_loss.item():.4f}, Aux: {aux_loss.item():.4f})")


if __name__ == "__main__":
    # test_share_moe()
    test_moe_training()