import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import custom_wkv_kernel
    print("Successfully imported custom_wkv_kernel.")
    use_custom_kernel = True
except ImportError:
    print("Failed to import custom_wkv_kernel. Falling back to PyTorch implementation.")
    use_custom_kernel = False


class WKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_head, w_head, k_bar_head, v_head, kappa_hat_head, a_head, initial_wkv_state):
        # The forward pass is likely fine as autocast handles the input types.
        # It is good practice to ensure inputs are contiguous.
        r_head = r_head.contiguous()
        w_head = w_head.contiguous()
        k_bar_head = k_bar_head.contiguous()
        v_head = v_head.contiguous()
        kappa_hat_head = kappa_hat_head.contiguous()
        a_head = a_head.contiguous()
        initial_wkv_state = initial_wkv_state.contiguous()
        
        # We save the original (potentially float16) tensors for memory efficiency.
        # We will cast them to float32 only when needed in the backward pass.
        ctx.save_for_backward(r_head, w_head, k_bar_head, v_head, kappa_hat_head, a_head, initial_wkv_state)
        
        # The custom kernel's forward pass MUST be able to handle the autocast dtype (e.g., float16)
        # Or we would need to cast inputs here as well. Let's assume the forward kernel is robust.
        outputs = custom_wkv_kernel.forward(r_head, w_head, k_bar_head, v_head, kappa_hat_head, a_head, initial_wkv_state)
        
        p_prime_all_steps, final_wkv_state, state_cache = outputs[0], outputs[1], outputs[2]
        
        # The state_cache is created inside the kernel, its dtype matters.
        # Let's save it along with other tensors.
        ctx.state_cache = state_cache
        
        return p_prime_all_steps, final_wkv_state

    @staticmethod
    def backward(ctx, grad_p_prime_all_steps, grad_final_wkv_state):
        # grad_p_prime_all_steps and grad_final_wkv_state are incoming gradients.
        # Autograd engine ensures they have the correct dtype matching the output of forward.
        # However, the custom kernel expects float32.
        
        # Retrieve saved tensors. They might be float16.
        r_head, w_head, k_bar_head, v_head, kappa_hat_head, a_head, initial_wkv_state = ctx.saved_tensors
        state_cache = ctx.state_cache
        
        # --- CRITICAL FIX ---
        # Explicitly cast all tensors passed to the custom C++/CUDA backward function to float32.
        grads = custom_wkv_kernel.backward(
            grad_p_prime_all_steps.contiguous().float(), 
            grad_final_wkv_state.contiguous().float(),
            r_head.contiguous().float(), 
            w_head.contiguous().float(), 
            k_bar_head.contiguous().float(), 
            v_head.contiguous().float(), 
            kappa_hat_head.contiguous().float(), 
            a_head.contiguous().float(),
            state_cache.contiguous().float()
        )
        # The returned grads from the kernel should be float32. Autograd will handle casting
        # them back to float16 if required for the rest of the backward pass.
        
        # Order of returned grads: grad_r, grad_w, grad_k_bar, grad_v, grad_kappa_hat, grad_a, grad_initial_wkv_state
        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6]

class LoraMLP(nn.Module):
    """
    LoRA MLP (Low-Rank Adaptation Multi-Layer Perceptron) as described in the RWKV-7 paper.
    This is a 2-layer MLP with a smaller hidden dimension (d_lora_hidden) compared to
    the input and output dimensions (d_model).
    It's used to implement data dependency with minimal parameters (see eq.2 in paper).
    """
    def __init__(self, d_model, d_lora_hidden, bias=True, activation_fn=None):
        super().__init__()
        # First linear layer: projects from d_model to d_lora_hidden
        self.fc1 = nn.Linear(d_model, d_lora_hidden, bias=bias)
        # Second linear layer: projects from d_lora_hidden back to d_model
        self.fc2 = nn.Linear(d_lora_hidden, d_model, bias=bias)
        # Optional activation function to apply after the first linear layer
        self.activation_fn = activation_fn

    def forward(self, x):
        # Pass input through the first linear layer
        h = self.fc1(x)
        # Apply activation function if provided
        if self.activation_fn:
            h = self.activation_fn(h)
        # Pass through the second linear layer and return
        return self.fc2(h)

class RWKV_TimeMix(nn.Module):
    """
    RWKV-7 Time Mixing Block.
    This block is responsible for mixing information across the time dimension.
    It implements the core WKV (Weighted Key Value) state evolution mechanism.
    Corresponds to Section 4.1 and Figure 1 (Time Mix part) in the RWKV-7 paper.
    """
    def __init__(self, d_model, head_size, num_heads, layer_id,
                 lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g):
        super().__init__()
        self.d_model = d_model  # C: Model dimension
        self.head_size = head_size # N: Dimension of each head
        self.num_heads = num_heads # H: Number of heads
        self.layer_id = layer_id # Current layer index, used for special handling of layer 0

        # Learnable token shift interpolation factors (mu parameters, eq.3)
        # These control the mix between current and previous token's representation
        # for generating various components (r, k, v, d, a, g).
        self.mu_r = nn.Parameter(torch.rand(d_model)) # For receptance (r)
        self.mu_k = nn.Parameter(torch.rand(d_model)) # For key (k)
        self.mu_v = nn.Parameter(torch.rand(d_model)) # For value (v) precursor
        self.mu_d = nn.Parameter(torch.rand(d_model)) # For decay (w) precursor (d_t)
        self.mu_a = nn.Parameter(torch.rand(d_model)) # For in-context learning rate (a) precursor
        self.mu_g = nn.Parameter(torch.rand(d_model)) # For RWKV gate (g)

        # Linear projections for Receptance (r), Key (k), and Value (v) precursors
        # These are W_r, W_k, W_v in eq. 5, 9, 13
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v_current_layer = nn.Linear(d_model, d_model, bias=False) # W_v for v_t,l (current layer value precursor)

        # LoRA MLPs for data-dependent parameters (d, a, v_gate, g)
        # These implement the loramlp_*(...) functions in eq. 4, 8, 11, 14
        self.decay_lora = LoraMLP(d_model, lora_dim_w, bias=True, activation_fn=torch.tanh) # For d_t (decay precursor, eq.11)
        self.iclr_lora = LoraMLP(d_model, lora_dim_a, bias=True) # For a_t (in-context learning rate, eq.4), pre-sigmoid
        self.value_residual_gate_lora = LoraMLP(d_model, lora_dim_v, bias=True) # For value residual gate (eq.8), pre-sigmoid
        self.gate_lora = LoraMLP(d_model, lora_dim_g, bias=False, activation_fn=torch.sigmoid) # For g_t (RWKV gate, eq.14)

        # Learnable parameters for key modifications
        # xi: removal key multiplier (eq.6)
        self.removal_key_multiplier_xi = nn.Parameter(torch.randn(d_model))
        # alpha: replacement rate booster for k_bar_t (eq.7) - the most right term
        self.iclr_mix_alpha = nn.Parameter(torch.rand(d_model))

        # Learnable parameter rho for WKV Bonus (eq.20), per head
        self.bonus_multiplier_rho = nn.Parameter(torch.randn(num_heads, self.head_size))

        # Output projection W_o (eq.22)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        # GroupNorm after WKV readout (p_t in eq.21, also see Fig 11 / pseudocode in Appendix G)
        self.ln_out_tm = nn.GroupNorm(num_heads, d_model)

    def forward(self, x, v_prime_c, shift_state_prev, wkv_state_prev):
        # x: input tensor (Batch, Time, Channels)
        # v_prime_c: value precursor from layer 0 (x_emb_shifted @ W_v_emb_level), used for value residual learning (eq.10)
        # shift_state_prev: previous token's representation for token shift (B, 1, C)
        # wkv_state_prev: previous WKV state (B, H, N, N)

        B, T, C = x.shape # Batch size, Sequence length, Model dimension
        H = self.num_heads # Number of heads
        N = self.head_size  # Head dimension

        # Token shift: x_shifted is x_{t-1} (eq.3)
        # Initialize shift_state_prev if it's the beginning of a sequence
        if shift_state_prev is None:
            shift_state_prev = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        # Concatenate previous shift state with current inputs (excluding the last token)
        # See Pseudocode in Appendix G, Line 11-12
        x_shifted = torch.cat([shift_state_prev, x[:, :-1]], dim=1)
        # Save the last token's representation as the new shift state for the next call
        current_shift_state = x[:, -1:, :].clone()

        # Interpolated inputs using lerp(x_t, x_{t-1}, mu) = x_t + (x_{t-1} - x_t) * mu (eq.3)
        # This is equivalent to (1-mu)*x_t + mu*x_{t-1}
        x_r_lerp = x + (x_shifted - x) * self.mu_r
        x_k_lerp = x + (x_shifted - x) * self.mu_k
        x_v_lerp = x + (x_shifted - x) * self.mu_v
        x_d_lerp = x + (x_shifted - x) * self.mu_d
        x_a_lerp = x + (x_shifted - x) * self.mu_a
        x_g_lerp = x + (x_shifted - x) * self.mu_g

        # --- Weight Preparation (Section 4.1 in paper) ---

        # Receptance (r_t), Key precursor (k_t), Value precursor (v_t,l) (eq. 5, 9, 13)
        r_vec = self.W_r(x_r_lerp)    # (B,T,C) - eq.13
        k_vec = self.W_k(x_k_lerp)    # (B,T,C) - eq.5 (key precursor)
        v_prime_l = self.W_v_current_layer(x_v_lerp) # v_t,l (current layer value precursor, eq.9)

        # Decay (w_t) (eq. 11, 12)
        d_lora_out = self.decay_lora(x_d_lerp) # Output of loramlp_d for d_t (eq.11)
        # w_t = exp(-e^{-0.5} * sigmoid(d_t)) (eq.12)
        # Using .float() for sigmoid and exp for stability, then casting back.
        w_vec = torch.exp(-torch.exp(torch.tensor(-0.5, device=x.device, dtype=torch.float32)) * torch.sigmoid(d_lora_out.float())).type_as(x)

        # In-context Learning Rate (a_t) and RWKV Gate (g_t)
        a_vec = torch.sigmoid(self.iclr_lora(x_a_lerp).float()).type_as(x) # a_t (eq.4)
        g_vec = self.gate_lora(x_g_lerp) # g_t (eq.14), sigmoid is inside LoraMLP

        # Value (v_t) computation with residual learning (eq. 8, 10)
        # v_prime_c is the value precursor from layer 0 (x_emb_shifted @ W_v_emb_level), or current layer's x_v_lerp if layer_id is 0.
        # The paper states v_t,c and v_t,l. v_t,c is from layer 0, v_t,l is from current layer.
        # For layer_id == 0, v_t,c is effectively v_prime_l of this layer, so lerp becomes v_prime_l.
        # This implementation detail: if layer_id == 0, _v_prime_c_to_use becomes v_prime_l of this layer.
        # So, v_vec = v_prime_l + (v_prime_l - v_prime_l) * gate = v_prime_l.
        # If layer_id > 0, _v_prime_c_to_use is the v_prime_c passed from the embedding level.
        _v_prime_c_to_use = v_prime_l if self.layer_id == 0 else v_prime_c

        value_residual_gate = torch.sigmoid(self.value_residual_gate_lora(x_v_lerp).float()).type_as(x) # Gate for lerp (eq.8)
        # v_t = lerp(v_t,c, v_t,l, gate) = v_t,c + (v_t,l - v_t,c) * gate (eq.10)
        v_vec = _v_prime_c_to_use + (v_prime_l - _v_prime_c_to_use) * value_residual_gate

        # Modified Keys: Removal Key (kappa_t) and Replacement Key (k_bar_t)
        kappa_vec = k_vec * self.removal_key_multiplier_xi # kappa_t = k_t * xi (eq.6)
        # k_bar_t = k_t * lerp(1, a_t, alpha) = k_t * (1 + (a_t - 1) * alpha) (eq.7)
        # This is equivalent to k_t * ( (1-alpha)*1 + alpha*a_t )
        k_bar_vec = k_vec * (1 + (a_vec - 1) * self.iclr_mix_alpha)

        # --- Reshape for multi-head operations ---
        # All these vectors are (B, T, C). Reshape to (B, T, H, N).
        r_head = r_vec.view(B, T, H, N)
        w_head = w_vec.view(B, T, H, N)         # Decay per head
        k_bar_head = k_bar_vec.view(B, T, H, N) # Replacement key per head
        v_head = v_vec.view(B, T, H, N)         # Value per head
        kappa_head = kappa_vec.view(B, T, H, N) # Removal key precursor per head
        a_head = a_vec.view(B, T, H, N)         # ICLR per head

        # Normalized removal key (kappa_hat_t) (eq.15)
        # Normalization is done per head over the N dimension.
        kappa_hat_head = F.normalize(kappa_head, p=2, dim=-1)

        # --- WKV state evolution (Recurrence) (eq.16, 17) ---
        # Initialize WKV state if it's the first pass
        if wkv_state_prev is None:
            wkv_state_prev = torch.zeros(B, H, N, N, device=x.device, dtype=x.dtype) # (B,H,N,N) matrix per head

        if use_custom_kernel:
            # Use the custom CUDA kernel
            # WKVFunction.forward is expected to return (p_prime_all_steps, final_wkv_state)
            # These are then returned by WKVFunction.apply
            applied_outputs = WKVFunction.apply(
                r_head, w_head, k_bar_head, v_head, kappa_hat_head, a_head, wkv_state_prev
            )

            if not (isinstance(applied_outputs, tuple) and len(applied_outputs) == 2 and
                    isinstance(applied_outputs[0], torch.Tensor) and
                    isinstance(applied_outputs[1], torch.Tensor)):
                # This will catch if applied_outputs is None or not the expected tuple of 2 tensors
                raise RuntimeError(
                    f"WKVFunction.apply(...) returned an unexpected value: {applied_outputs}. "
                    "This often indicates a crash or critical error within the custom C++/CUDA kernel "
                    "or the autograd.Function lifecycle. Expected a tuple of 2 tensors."
                )

            p_prime_stacked, final_wkv_state_to_pass = applied_outputs
            # p_prime_stacked is (B, T, H, N)
            p_prime = p_prime_stacked.view(B, T, C) # Reshape to (B,T,C)
        else:
            wkv_readouts_over_time = [] # To store the readout from WKV state at each time step
            current_wkv_state = wkv_state_prev # This is wkv_{t-1}

            # Iterate over each time step in the sequence
            for t_step in range(T):
                # Get per-timestep components (B,H,N)
                rt, wt, kt_bar, vt, kappat_hat, at = \
                    r_head[:,t_step], w_head[:,t_step], k_bar_head[:,t_step], v_head[:,t_step], \
                    kappa_hat_head[:,t_step], a_head[:,t_step]

                # Calculate Transition Matrix G_t (eq.19)
                # G_t = diag(w_t) - kappa_hat_t^T (a_t . kappa_hat_t)
                # (a_t . kappa_hat_t) is element-wise product
                term_inside_paren = at * kappat_hat # (B,H,N)
                # Outer product: kappa_hat_t (column vector) * term_inside_paren (row vector)
                # kappat_hat.unsqueeze(-1) gives (B,H,N,1)
                # term_inside_paren.unsqueeze(-2) gives (B,H,1,N)
                outer_prod_term = kappat_hat.unsqueeze(-1) * term_inside_paren.unsqueeze(-2) # (B,H,N,N)
                diag_wt = torch.diag_embed(wt) # Creates diagonal matrices (B,H,N,N) from wt (B,H,N)
                G_t = diag_wt - outer_prod_term # (B,H,N,N)

                # Calculate Additive term: v_t^T . k_bar_t (outer product) (eq.17)
                # vt.unsqueeze(-1) gives (B,H,N,1)
                # kt_bar.unsqueeze(-2) gives (B,H,1,N)
                vk_outer_prod = vt.unsqueeze(-1) * kt_bar.unsqueeze(-2) # (B,H,N,N)

                # WKV state update: wkv_t = wkv_{t-1} @ G_t + v_t^T . k_bar_t (eq.17)
                # Note: paper uses row vectors, so state update is S_t = S_{t-1} * G_t + ...
                # Here, if wkv_state is D_h x D_h, and G_t is D_h x D_h, then wkv_state @ G_t is correct.
                current_wkv_state = current_wkv_state @ G_t + vk_outer_prod # (B,H,N,N)

                # Readout from WKV state: r_t @ wkv_t^T (as per pseudocode in Appendix G, line 53: y = wkv_state @ r_t)
                # The paper's math (eq.21) has r_t * wkv_t^T.
                # If r_t is (B,H,N) and wkv_state is (B,H,N,N), then r_t @ wkv_state^T would be:
                # (B,H,N) @ (B,H,N,N) -> (B,H,N).
                # The einsum 'bhn,bhmn->bhm' means r_t (row vector) times wkv_state (matrix).
                # This matches r_t * W where W is the wkv_state.
                wkv_readout_t = torch.einsum('bhn,bhmn->bhm', rt, current_wkv_state) # (B,H,N)
                wkv_readouts_over_time.append(wkv_readout_t)

            # The final WKV state after processing all T steps
            final_wkv_state_to_pass = current_wkv_state.clone()

            # Stack readouts from all time steps
            # p_prime is the result of (r_t wkv_t^T) from eq.21, before LayerNorm and bonus
            p_prime = torch.stack(wkv_readouts_over_time, dim=1) # (B, T, H, N)
            p_prime = p_prime.view(B, T, C) # Reshape to (B,T,C)

        # Normalize p_prime using GroupNorm (eq.21, and Fig 11)
        # GroupNorm expects (B, C, ...) input, so transpose T and C
        p_prime_norm = self.ln_out_tm(p_prime.transpose(1,2).contiguous()).transpose(1,2).contiguous()

        # WKV Bonus u_t (eq.20)
        # u_t = (r_t . (rho . k_bar_t)^T) v_t
        # Interpretation:
        # 1. rho . k_bar_t : element-wise product (B,T,H,N)
        # 2. r_t . (result_of_1) : inner product (sum over N), resulting in a scalar per head (B,T,H,1)
        # 3. scalar_per_head * v_t : element-wise product (B,T,H,N)
        rho_expanded = self.bonus_multiplier_rho.unsqueeze(0).unsqueeze(0) # (1,1,H,N) to allow broadcasting
        term_rho_k_bar = rho_expanded * k_bar_head # (B,T,H,N)
        # Inner product for each head: (r_head * term_rho_k_bar) summed over the N dimension
        scalar_per_head = (r_head * term_rho_k_bar).sum(dim=-1, keepdim=True) # (B,T,H,1)
        bonus_u_head = scalar_per_head * v_head # (B,T,H,N)
        bonus_u = bonus_u_head.view(B,T,C) # Reshape to (B,T,C)

        # Add bonus to normalized readout (eq.21, p_t = LayerNorm(...) + u_t)
        p_t = p_prime_norm + bonus_u # (B,T,C)

        # Final gating and output projection (eq.22)
        # o_t = (g_t . p_t) W_o
        # g_vec is (B,T,C)
        output = self.W_o(g_vec * p_t)

        return output, current_shift_state, final_wkv_state_to_pass


class RWKV_ChannelMix(nn.Module):
    """
    RWKV-7 Channel Mixing Block (Modified with SwiGLU).
    This block mixes information across channels (the model dimension C).
    It's a two-layer MLP with SwiGLU activation instead of ReLU squared.
    Modified from Section 4.2 and Figure 1 (ReLU^2 MLP part) in the RWKV-7 paper.
    """
    def __init__(self, d_model, hidden_dim_multiplier=4):
        super().__init__()
        self.d_model = d_model # C
        self.hidden_dim = d_model * hidden_dim_multiplier # Hidden dimension of the MLP (4*C in paper)        # Token shift interpolation factor (mu_k_prime) for k_t_prime (eq.23)
        self.mu_k_prime = nn.Parameter(torch.rand(d_model))

        # Linear projections for the MLP (eq.23, 24)
        self.W_k_prime = nn.Linear(d_model, self.hidden_dim, bias=False) # Projects to hidden dim
        # Output projection
        self.W_v_prime = nn.Linear(self.hidden_dim, d_model, bias=False) # Projects back to model dim

    def forward(self, x, shift_state_prev):
        # x: input tensor (B, T, C)
        # shift_state_prev: previous token's representation for token shift (B, 1, C)
        B, T, C = x.shape

        # Token shift for k_t_prime (eq.23)
        if shift_state_prev is None:
            shift_state_prev = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        x_shifted = torch.cat([shift_state_prev, x[:, :-1]], dim=1)
        current_shift_state = x[:, -1:, :].clone() # Save last token for next step

        # Interpolated input for k_t_prime: lerp(x_t, x_{t-1}, mu_k_prime) (eq.23)
        x_k_prime_lerp = x + (x_shifted - x) * self.mu_k_prime
        
        # k_t_prime = (lerped_input) @ W_k_prime (eq.23)
        k_prime = self.W_k_prime(x_k_prime_lerp) # (B, T, hidden_dim)

        # Output: o_t_prime = ReLU(k_t_prime)^2 @ W_v_prime (eq.24)
        output = self.W_v_prime(torch.relu(k_prime)**2) # (B, T, C)
        return output, current_shift_state


class RWKV_Block(nn.Module):
    """
    A single RWKV-7 Block, combining a TimeMix block and a ChannelMix block.
    Includes RMSNorms before each sub-block and residual connections (modified from LayerNorms).
    Corresponds to one layer (L_x) in Figure 1.
    """
    def __init__(self, d_model, head_size, num_heads, layer_id, ffn_hidden_multiplier,
                 lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g):
        super().__init__()
        self.layer_id = layer_id # To inform TimeMix about special handling for layer 0        
        # RMSNorm before TimeMix (changed from LayerNorm)
        self.ln_tm_in = nn.RMSNorm(d_model)
        # TimeMix block
        self.tm = RWKV_TimeMix(d_model, head_size, num_heads, layer_id,
                               lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g)        
        # RMSNorm before ChannelMix (changed from LayerNorm)
        self.ln_cm_in = nn.RMSNorm(d_model)
        # ChannelMix block
        self.cm = RWKV_ChannelMix(d_model, ffn_hidden_multiplier)

    def forward(self, x, v_prime_c, # Value precursor from layer 0, for TimeMix
                tm_shift_state_prev, tm_wkv_state_prev, cm_shift_state_prev):
        # x: input from previous block or embedding (B,T,C)
        # v_prime_c: value precursor from embedding level, passed to TimeMix
        # tm_shift_state_prev, tm_wkv_state_prev: states for TimeMix
        # cm_shift_state_prev: state for ChannelMix

        # --- Time Mixing part ---
        tm_input = self.ln_tm_in(x) # Apply LayerNorm
        dx_tm, next_tm_shift_state, next_tm_wkv_state = self.tm(
            tm_input, v_prime_c, tm_shift_state_prev, tm_wkv_state_prev
        )
        x = x + dx_tm # Residual connection

        # --- Channel Mixing part ---
        cm_input = self.ln_cm_in(x) # Apply LayerNorm
        dx_cm, next_cm_shift_state = self.cm(cm_input, cm_shift_state_prev)
        x = x + dx_cm # Residual connection

        return x, next_tm_shift_state, next_tm_wkv_state, next_cm_shift_state


class RWKV7_Model_Classifier(nn.Module):
    """
    Full RWKV-7 Model adapted for sequence classification.
    This class stacks multiple RWKV_Blocks and adds an embedding layer at the beginning
    and a classification head at the end.
    """
    def __init__(self, d_model, n_layer, vocab_size, head_size=64, ffn_hidden_multiplier=4, # Default head_size to 64 as in paper
                 lora_dim_w=64, lora_dim_a=64, lora_dim_v=32, lora_dim_g=128): # Example LoRA dims
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.head_size = head_size
        if d_model % head_size != 0:
            raise ValueError("d_model must be divisible by head_size for RWKV TimeMix.")
        self.num_heads = d_model // head_size
        self.vocab_size = vocab_size

        # Input Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Parameters for v_prime_c (value precursor from layer 0, used in all TimeMix blocks)
        # This is v_{t,c} in eq.10, derived from the initial embedding.
        # It needs its own token shift and projection.
        self.mu_v_for_v_prime_c = nn.Parameter(torch.rand(d_model)) # Token shift mu for v_prime_c input
        self.W_v_for_v_prime_c = nn.Linear(d_model, d_model, bias=False) # Projection W_v for v_prime_c        
        # RMSNorm after embedding, before the first block (changed from LayerNorm)
        self.ln_pre_blocks = nn.RMSNorm(d_model)

        # Stack of RWKV_Blocks
        self.blocks = nn.ModuleList([
            RWKV_Block(d_model, head_size, self.num_heads, i, ffn_hidden_multiplier,
                       lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g)
            for i in range(n_layer)
        ])        
        # Final RMSNorm before the classification head (changed from LayerNorm)
        self.ln_post_blocks = nn.RMSNorm(d_model)
        # Classification head (e.g., for binary classification, outputting a single logit)
        self.classification_head = nn.Linear(d_model, 1) # Example: 1 output for binary classification

    def forward(self, input_ids, states_list_prev=None):
        # input_ids: (Batch, Time) tensor of token IDs
        # states_list_prev: Optional list of dictionaries, each containing states for a block
        #                   [{'tm_shift_state': ..., 'tm_wkv_state': ..., 'cm_shift_state': ...}, ...]

        B, T = input_ids.shape

        # If sequence length is 0, handle appropriately (e.g., return zeros or handle in loss)
        # This check was in the original code, might be for specific use cases.
        if T == 0:
            # Depending on expected output shape, might need to return something specific.
            # For now, assume T > 0 or this case is handled by the caller.
            # If we need to produce a (B,1) logit, we'd need a default.
            # For simplicity, let's assume T > 0 for now.
            # If a state needs to be returned, it should match the expected structure.
            dummy_logits = torch.zeros(B, 1, device=input_ids.device) # Example dummy output
            
            # Construct dummy states if needed for consistency
            dummy_states_list = []
            if states_list_prev is None: # If no previous states, create new zero states
                 for _ in range(self.n_layer):
                    dummy_states_list.append({
                        'tm_shift_state': torch.zeros(B, 1, self.d_model, device=input_ids.device, dtype=self.embedding.weight.dtype),
                        'tm_wkv_state': torch.zeros(B, self.num_heads, self.head_size, self.head_size, device=input_ids.device, dtype=self.embedding.weight.dtype),
                        'cm_shift_state': torch.zeros(B, 1, self.d_model, device=input_ids.device, dtype=self.embedding.weight.dtype)
                    })
            else: # If previous states exist, return them (or updated versions if T=0 implies some state update)
                dummy_states_list = states_list_prev

            return dummy_logits, dummy_states_list


        # 1. Input Embedding
        x_emb = self.embedding(input_ids) # (B, T, C)

        # 2. Calculate v_prime_c (value precursor from layer 0)
        # This v_prime_c is token-specific (varies along T) and is passed to all TimeMix blocks.
        # It's derived from the initial token-shifted embeddings.
        # Initial shift state for v_prime_c calculation (always zeros at the start of this forward pass for v_prime_c)
        initial_shift_state_for_vpc = torch.zeros(B, 1, self.d_model, device=x_emb.device, dtype=x_emb.dtype)
        x_emb_shifted_for_vpc = torch.cat([initial_shift_state_for_vpc, x_emb[:, :-1]], dim=1) # x_{t-1} for embedding
        # Lerp input for v_prime_c
        x_v_lerp_for_vpc = x_emb + (x_emb_shifted_for_vpc - x_emb) * self.mu_v_for_v_prime_c
        v_prime_c = self.W_v_for_v_prime_c(x_v_lerp_for_vpc) # (B, T, C)

        # 3. Initial LayerNorm after embedding
        current_x = self.ln_pre_blocks(x_emb)

        # 4. Initialize states if not provided (e.g., for the first chunk of a long sequence)
        if states_list_prev is None:
            states_list_prev = []
            for _ in range(self.n_layer):
                states_list_prev.append({
                    # TimeMix shift state: (B, 1, C) - stores x_{T} of previous chunk
                    'tm_shift_state': torch.zeros(B, 1, self.d_model, device=current_x.device, dtype=current_x.dtype),
                    # TimeMix WKV state: (B, H, N, N) - stores wkv_{T} of previous chunk
                    'tm_wkv_state': torch.zeros(B, self.num_heads, self.head_size, self.head_size, device=current_x.device, dtype=current_x.dtype),
                    # ChannelMix shift state: (B, 1, C) - stores x_{T} of previous chunk (after TimeMix)
                    'cm_shift_state': torch.zeros(B, 1, self.d_model, device=current_x.device, dtype=current_x.dtype)
                })

        next_states_list_to_return = [] # To store the updated states from each block

        # 5. Pass through RWKV Blocks
        for i in range(self.n_layer):
            block_state_prev = states_list_prev[i]
            current_x, next_tm_ss, next_tm_ws, next_cm_ss = self.blocks[i](
                current_x, v_prime_c, # Pass the same v_prime_c (from initial embedding) to all layers
                block_state_prev['tm_shift_state'],
                block_state_prev['tm_wkv_state'],
                block_state_prev['cm_shift_state']
            )
            next_states_list_to_return.append({
                'tm_shift_state': next_tm_ss,
                'tm_wkv_state': next_tm_ws,
                'cm_shift_state': next_cm_ss
            })

        # 6. Final LayerNorm after all blocks
        final_x_representation = self.ln_post_blocks(current_x) # (B, T, C)

        # 7. Classification Head
        # For sequence classification, typically the representation of the last token (or a CLS token) is used.
        # Here, we use the hidden state of the last actual token.
        # If input_ids are padded, a more robust way is to gather the state of the last non-padding token.
        # Assuming for now that the last token at T-1 is the one to use.
        last_token_hidden_state = final_x_representation[:, -1, :] # (B, C)

        class_logits = self.classification_head(last_token_hidden_state) # (B, num_classes), here (B, 1)

        return class_logits, next_states_list_to_return
