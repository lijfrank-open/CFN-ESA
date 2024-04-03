import torch
from torch import nn
from torch.nn import functional as F

class EmoShift(nn.Module):
    def __init__(self, d_model, output_dim=128, dropout=0.9, diff_type="concat"):
        super().__init__()
        self.diff_type = diff_type
        diff_hidden_dim = 2*d_model
        self.fc = nn.Sequential(
        		nn.Linear(diff_hidden_dim, output_dim),
        		nn.ReLU(),
        		nn.Dropout(dropout),
        	)
        self.classify = nn.Sequential(
                nn.Linear(output_dim, 2),
            )

    def _build_match_sample(self, embeds, umask, qmask, embeds_contrastive=None):
        if embeds_contrastive == None:
            embeds_contrastive = embeds.clone()
        elif self.diff_type == "concat":
            seq_len = embeds.shape[0] 
            embeds_diff = torch.cat([embeds[:, None].repeat(1,seq_len,1,1), 
                                    embeds_contrastive[None, :].repeat(seq_len,1,1,1)], dim=-1)
        else:
            raise TypeError("the diff_type belongs to [\"concat\"]")
        return embeds_diff

    def forward(self, embeds, umask, qmask, embeds_cmp=None):
        embeds_fusion = embeds
        embeds_contrastive = None if embeds_cmp==None else embeds_cmp
        embeds_diff = self._build_match_sample(embeds_fusion, umask, qmask, embeds_contrastive=embeds_contrastive)
        embeds_fc = self.fc(embeds_diff)
        logits = self.classify(embeds_fc)
        return logits