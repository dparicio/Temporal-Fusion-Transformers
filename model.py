import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GLU(nn.Module):
    """Gated Linear Unit (GLU)"""
    # Inputs can be of shape [B, H] or [B, T, H]
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size, hidden_size * 2, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)

        # Split into activations and gates (pre-sigmoid)
        a, gate_pre = torch.split(x, self.hidden_size, dim=-1)

        gate = torch.sigmoid(gate_pre)
        y = a * gate
        return y, gate
    

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0):
        super().__init__()
        output_size = output_size if output_size is not None else hidden_size

        # Dense layers main path (same notation as in paper)
        self.fc2 = nn.Linear(input_size, hidden_size, bias=True) 
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False) # for context
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)

        self.glu = GLU(input_size=hidden_size, hidden_size=output_size, dropout=dropout)
        self.elu = nn.ELU()
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size, bias=False)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, context=None, return_gate=False):
        x0 = x  # for skip connection

        # Compute nu2
        x = self.fc2(x)
        if context is not None:
            x = x + self.fc3(context)
        x = self.elu(x)

        # Compute nu1
        x = self.fc1(x)

        # Compute GRN as GLU + skip + layer norm
        y, gate = self.glu(x)
        out = self.layer_norm(self.skip(x0) + y)

        # Returns the GLU gate for diagnostic purposes (in the paper)
        return (out, gate) if return_gate else out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.0):
        super().__init__()
        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        # Compute scaled dot-product attention
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        attn = self.activation(scores)
        attn = self.dropout(attn)
        # Compute attention with sharing values
        output = torch.matmul(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_model // n_head

        self.qs_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.ks_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.vs_layer = nn.Linear(d_model, self.d_v, bias=False)

        self.attention = ScaledDotProductAttention(attn_dropout=0.0)
        self.head_dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(self.d_v, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layer(v)
            head, attn = self.attention(qs, ks, vs, attn_mask)
            head = self.head_dropout(head)
            heads.append(head)
            attns.append(attn)

        attn_stack = torch.stack(attns, dim=0)
        # Combine heads by averaging for interpretable attention
        if self.n_head > 1:
            head_stack = torch.stack(heads, dim=0)
            outputs = head_stack.mean(dim=0)
        else:
            outputs = heads[0]

        outputs = self.w_o(outputs)
        outputs = self.out_dropout(outputs)

        return outputs, attn_stack


class TemporalFusionTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.enc_len = params['encoder_length']
        self.dec_len = params['decoder_length']
        self.time_steps = params['time_steps']
        self.feature_description = params['feature_description']
        # Map number of embeddings per categorical variable
        self.embed_per_cat = params['embed_per_cat']
        # Hidden state size (common across TFT)
        self.d_model = params['d_model']
        self.dropout = params['dropout']
        self.n_head  = params['n_head']
        self.quantiles = params['quantiles']

        # Input features
        self.static_categorical_inputs = len(self.feature_description.static_categorical)
        self.static_continuous_inputs = len(self.feature_description.static_continuous)
        self.known_categorical_inputs = len(self.feature_description.known_categorical)
        self.known_continuous_inputs = len(self.feature_description.known_continuous)
        self.obs_categorical_inputs = len(self.feature_description.observed_categorical)
        self.obs_continuous_inputs = len(self.feature_description.observed_continuous)
        self.n_cat = (self.static_categorical_inputs + 
                      self.known_categorical_inputs + 
                      self.obs_categorical_inputs)
        self.n_cont = (self.static_continuous_inputs + 
                       self.known_continuous_inputs + 
                       self.obs_continuous_inputs)

        # Embeddings
        self.linear_projections = nn.ModuleList([nn.Linear(1, self.d_model) for _ in range(self.n_cont)])
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.embed_per_cat[i], 
                                                      embedding_dim=self.d_model, 
                                                      padding_idx=0) for i in range(self.n_cat)])
        
        
        self.n_static = self.static_categorical_inputs + self.static_continuous_inputs
        if self.n_static > 0:
            # Static variable selection GRNs
            self.static_scorer = GRN(
                input_size=self.d_model * self.n_static,
                hidden_size=self.d_model,
                output_size=self.n_static,
                dropout=self.dropout
            )
            self.static_var_grns = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(self.n_static)
            ])

            # Static covariate encoders GRNs (4 in total)
            self.static_context_grns = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(4)
            ])

        # Temporal variable selection GRNs
        # Number of historical and future features
        self.n_hist = (self.obs_categorical_inputs + self.obs_continuous_inputs) + (self.known_categorical_inputs + self.known_continuous_inputs)
        self.n_fut  = (self.known_categorical_inputs + self.known_continuous_inputs)

        if self.n_hist > 0:
            self.temporal_scorer_hist = GRN(
                input_size=self.d_model * self.n_hist,   
                hidden_size=self.d_model,
                output_size=self.n_hist,                 
                dropout=self.dropout
            )
            self.temporal_var_grns_hist = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(self.n_hist)              
            ])

        if self.n_fut > 0:
            self.temporal_scorer_fut = GRN(
                input_size=self.d_model * self.n_fut,    
                hidden_size=self.d_model,
                output_size=self.n_fut,                  
                dropout=self.dropout
            )
            self.temporal_var_grns_fut = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(self.n_fut)               
            ])

        # Define LSTM encoder and decoder (NOTE: 1 layer only)
        self.lstm_enc = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.lstm_dec = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Local enhancement layer (GLU and LayerNorm)
        self.temporal_gate = GLU(input_size=self.d_model, hidden_size=self.d_model, dropout=self.dropout)
        self.temporal_ln   = nn.LayerNorm(self.d_model)

        # Static enrichment GRN
        self.static_enrichment_grn = GRN(
            input_size=self.d_model,
            hidden_size=self.d_model,
            output_size=self.d_model,
            dropout=self.dropout
        )

        # Temporal self-attention
        self.attention = InterpretableMultiHeadAttention(
            n_head=self.n_head,
            d_model=self.d_model,
            dropout=self.dropout
        )

        self.attention_gate = GLU(input_size=self.d_model, hidden_size=self.d_model, dropout=self.dropout)
        self.attention_ln   = nn.LayerNorm(self.d_model)

        # Position-wise feedforward layer (GRN, GLU and LayerNorm)
        self.positionwise_grn = GRN(
            input_size=self.d_model,
            hidden_size=self.d_model,
            output_size=self.d_model,
            dropout=self.dropout
        )
        self.positionwise_gate = GLU(input_size=self.d_model, hidden_size=self.d_model, dropout=self.dropout)
        self.positionwise_ln   = nn.LayerNorm(self.d_model)

        # Quantile output layers
        self.quantile_fc = nn.Linear(self.d_model, len(self.quantiles))

    def input2embedding(
        self, 
        static_categorical_inputs, 
        static_continuous_inputs, 
        known_categorical_inputs, 
        known_continuous_inputs, 
        obs_categorical_inputs, 
        obs_continuous_inputs
    ):
        n_static_cat = self.static_categorical_inputs
        n_static_cont = self.static_continuous_inputs
        n_known_cat = self.known_categorical_inputs
        n_known_cont = self.known_continuous_inputs
        n_obs_cat = self.obs_categorical_inputs
        n_obs_cont = self.obs_continuous_inputs

        # == CATEGORICAL VARIABLES ==
        static_cat_embeddings = []
        for i in range(n_static_cat):
            static_cat_embeddings.append(self.embeddings[i](static_categorical_inputs[..., i]))

        obs_cat_embeddings = []
        for i in range(n_obs_cat):
            obs_cat_embeddings.append(self.embeddings[n_static_cat + i](obs_categorical_inputs[..., i]))

        known_cat_embeddings = []
        for i in range(n_known_cat):
            known_cat_embeddings.append(self.embeddings[n_static_cat + n_obs_cat + i](known_categorical_inputs[..., i]))

        # == CONTINUOUS VARIABLES ==
        static_cont_embeddings = []
        for i in range(n_static_cont):
            static_cont_embeddings.append(self.linear_projections[i](static_continuous_inputs[..., i:i+1]))    

        obs_cont_embeddings = []
        for i in range(n_obs_cont):
            obs_cont_embeddings.append(self.linear_projections[n_static_cont + i](obs_continuous_inputs[..., i:i+1]))

        known_cont_embeddings = []
        for i in range(n_known_cont):
            known_cont_embeddings.append(self.linear_projections[n_static_cont + n_obs_cont + i](known_continuous_inputs[..., i:i+1]))

        # Stack all embeddings along a new dimension
        static_embeddings = None
        obs_embeddings    = None
        known_embeddings  = None

        if static_cat_embeddings or static_cont_embeddings:
            static_embeddings = torch.stack(static_cat_embeddings + static_cont_embeddings, dim=-1) # [B, H, n_static]
        if obs_cat_embeddings or obs_cont_embeddings:
            obs_embeddings    = torch.stack(obs_cat_embeddings + obs_cont_embeddings, dim=-1) # [B, T_enc, H, n_obs]
        if known_cat_embeddings or known_cont_embeddings:
            known_embeddings  = torch.stack(known_cat_embeddings + known_cont_embeddings, dim=-1) # [B, T_dec, H, n_known]

        return static_embeddings, obs_embeddings, known_embeddings


    def static_variable_selection(self, static_embeddings):
        #TODO: handled when no static inputs (static_embeddings is None)

        B, H, n_static = static_embeddings.shape

        # Flatten inputs
        flat = static_embeddings.reshape(B, H*n_static)

        # GRN to compute variable weights
        scorer = self.static_scorer
        static_weights = torch.softmax(scorer(flat), dim=-1).unsqueeze(1) # shape [B, 1, n_static]

        # Apply per-variable GRN
        transfomed_list = []
        for i in range(n_static):
            ti = self.static_var_grns[i](static_embeddings[:, :, i]).unsqueeze(-1)
            transfomed_list.append(ti)
        trans_static_embeddings = torch.cat(transfomed_list, dim=-1) # shape [B, H, n_static]

        # Apply weights to static embeddings
        static_vec = (static_weights * trans_static_embeddings).sum(dim=-1) # shape [B, H]

        return static_vec, static_weights


    def static_covariate_encoders(self, static_vec):
        """Computes the static covariate encoders"""
        static_context_variable_selection = self.static_context_grns[0](static_vec)
        static_context_enrichment = self.static_context_grns[1](static_vec)
        static_context_state_h = self.static_context_grns[2](static_vec)
        static_context_state_c = self.static_context_grns[3](static_vec)

        # shape [B, H]
        return (static_context_variable_selection, 
                static_context_enrichment, 
                static_context_state_h, 
                static_context_state_c)


    def temporal_variable_selection(
            self, 
            temporal_embeddings, 
            static_context_variable_selection,
            mode):
        
        # Retrieve correct scorer and var_grns
        if mode == 'hist':
            scorer = self.temporal_scorer_hist
            var_grns = self.temporal_var_grns_hist
        elif mode == 'fut':
            scorer = self.temporal_scorer_fut
            var_grns = self.temporal_var_grns_fut

        # Flatten inputs
        B, T, H, N = temporal_embeddings.shape
        flat = temporal_embeddings.reshape(B, T, H*N)

        # Convert static context to correct dimension
        # Context dim: [B, H] -> [B, 1, H] -> [B, T, H]
        context = static_context_variable_selection.unsqueeze(1).expand(B, T, H)

        # Compute variable weights (#NOTE return_gate=True like in paper)
        score, static_gate = scorer(flat, context=context, return_gate=True)  # shape: [B, T, N]
        temporal_weights = torch.softmax(score, dim=-1).unsqueeze(2)  # shape: [B, T, 1, N]

        # Apply per-variable GRN. shape [B, T, H, N]
        trans_temporal_embeddings = torch.stack([var_grns[i](temporal_embeddings[..., i]) for i in range(N)], dim=-1)

        # Apply weights to temporal embeddings. shape [B, T, H]
        temporal_vec = (temporal_weights * trans_temporal_embeddings).sum(dim=-1) 

        return temporal_vec, temporal_weights, static_gate
    

    def lstm_encoder_decoder(self, hist_inputs, fut_inputs, static_context_state_h, static_context_state_c):
        # Static context initialization shape: [B, H] -> [1, B, H]
        h0 = static_context_state_h.unsqueeze(0)   
        c0 = static_context_state_c.unsqueeze(0)  

        # LSTM encoder
        enc_out, (h, c) = self.lstm_enc(hist_inputs, (h0, c0))  

        # LSTM decoder
        dec_out, _ = self.lstm_dec(fut_inputs, (h, c))  

        lstm_layer = torch.cat([enc_out, dec_out], dim=1) 
        return lstm_layer
    

    def local_enhancement_layer(self, lstm_layer, input_embeddings):
        # Pass through final gating layer
        lstm_gated, _ = self.temporal_gate(lstm_layer)
    
        # Add and layer norm
        temporal_feature_layer = self.temporal_ln(lstm_gated + input_embeddings)

        return temporal_feature_layer


    def static_enrichment_layer(self, temporal_features, static_context_enrichment):
        # Expand static context to match temporal features shape
        context = static_context_enrichment.unsqueeze(1).expand(-1, temporal_features.shape[1], -1)
        # Apply static enrichment GRN
        enriched_static_context = self.static_enrichment_grn(temporal_features, context=context)
        return enriched_static_context
    
    
    def get_attention_mask(self, x):
        # Causal mask for attention
        # Prevents the self-attention layer from looking into future tokens when predicting
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
        return mask
    

    def temporal_self_attention_layer(self, enriched_temporal_layer):
        # Apply attention layer
        mask = self.get_attention_mask(enriched_temporal_layer)
        attention_output, _ = self.attention(enriched_temporal_layer, enriched_temporal_layer, enriched_temporal_layer, attn_mask=mask)

        attention_gated, _ = self.attention_gate(attention_output)
        attention_out = self.attention_ln(enriched_temporal_layer + attention_gated)

        return attention_out


    def positionwise_feedforward(self, temporal_feature_layer, attention_layer):
        positionwise_out      = self.positionwise_grn(attention_layer)
        positionwise_gated, _ = self.positionwise_gate(positionwise_out)
        transformer_layer     = self.positionwise_ln(temporal_feature_layer + positionwise_gated)
        return transformer_layer


    def quantile_output(self, transformer_layer):
        # Get decoder part
        decoder_layer = transformer_layer[:, self.enc_len:, :]

        # Apply quantile output layer (get predictions)
        yhat = self.quantile_fc(decoder_layer)
        return yhat


    def forward(self, batch):
        # Get inputs
        stat_cats = batch["model_inputs"]["static_cats"]
        stat_cont = batch["model_inputs"]["static_cont"]
        obs_cats  = batch["model_inputs"]["obs_cats"]
        obs_cont  = batch["model_inputs"]["obs_cont"]
        know_cats = batch["model_inputs"]["known_cats"]
        know_cont = batch["model_inputs"]["known_cont"]

        # Get embeddings shapes: [B, H, n_static], [B, T, H, n_obs], [B, T, H, n_known]
        static_embeddings, obs_embeddings, known_embeddings = self.input2embedding(
            static_categorical_inputs=stat_cats,
            static_continuous_inputs=stat_cont,
            known_categorical_inputs=know_cats,
            known_continuous_inputs=know_cont,
            obs_categorical_inputs=obs_cats,
            obs_continuous_inputs=obs_cont
        )  

        # Split temporal embeddings into historical and future
        if obs_embeddings is None:
            historical_inputs = known_embeddings[:, :self.enc_len, :, :]
        elif known_embeddings is None:
            historical_inputs = obs_embeddings[:, :self.enc_len, :, :]
        else:
            historical_inputs = torch.cat([obs_embeddings, known_embeddings[:, :self.enc_len, :, :]], dim=-1)
        future_inputs     = known_embeddings[:, self.enc_len:, :, :]     

        # Apply variable selection network to static inputs
        static_vec, static_weights = self.static_variable_selection(static_embeddings)

        # Get static covariate encoders
        static_context_variable_selection, static_context_enrichment, static_context_state_h, static_context_state_c = self.static_covariate_encoders(static_vec)

        # Apply temporal variable selection to historical and future inputs
        hist_features, hist_flags, _ = self.temporal_variable_selection(
            historical_inputs, static_context_variable_selection, mode="hist"
        )
        fut_features, fut_flags, _ = self.temporal_variable_selection(
            future_inputs, static_context_variable_selection, mode="fut"
        )

        # LSTM encoder-decoder
        lstm_layer = self.lstm_encoder_decoder(
            hist_features, 
            fut_features, 
            static_context_state_h, 
            static_context_state_c
        )

        # Create input embeddings for final gating layer
        input_embeddings = torch.cat([hist_features, fut_features], dim=1)

        # Apply local enhancement layer
        temporal_feature_layer = self.local_enhancement_layer(lstm_layer, input_embeddings)

        # Apply static enrichment layer
        enriched_temporal_layer = self.static_enrichment_layer(temporal_feature_layer, static_context_enrichment)

        # Apply temporal self-attention layer (attention + GLU + LayerNorm)
        attention_layer = self.temporal_self_attention_layer(enriched_temporal_layer)

        # Apply position-wise feedforward layer shape [B, T, H]
        transformer_layer = self.positionwise_feedforward(temporal_feature_layer, attention_layer)

        # Apply quantile output layer
        yhat = self.quantile_output(transformer_layer)

        return yhat


def quantile_loss(y_true, y_pred, quantiles):
    quantiles = torch.tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device) if not torch.is_tensor(quantiles) else quantiles.to(y_pred.device, y_pred.dtype) 
    errors = y_true - y_pred
    q = quantiles.view(1, 1, -1)
    loss = torch.maximum(q * errors, (q - 1) * errors)
    return loss.mean()