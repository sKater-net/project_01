import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedTernaryLoss(nn.Module):
    def __init__(self, margin1=0.5, margin2=0.8, alpha=0.5):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.alpha = alpha
        
    def forward(self, hash_img, hash_txt, labels):
        batch_size = hash_img.size(0)
        
        sim_i2t = hash_img.mm(hash_txt.t())
        sim_t2i = hash_txt.mm(hash_img.t())
        
        label_sim = labels.mm(labels.t()).float()
        
        loss_i2t = 0
        loss_t2i = 0
        
        for i in range(batch_size):
            pos_mask = label_sim[i] > 0
            neg_mask = ~pos_mask
            
            # Image to Text loss
            pos_sim = sim_i2t[i][pos_mask]
            neg_sim = sim_i2t[i][neg_mask]
            
            if len(pos_sim) > 0 and len(neg_sim) > 0:
                pos_loss = torch.relu(self.margin1 - pos_sim).mean()
                neg_loss = torch.relu(neg_sim - self.margin2 + self.margin1).mean()
                loss_i2t += pos_loss + self.alpha * neg_loss
            
            # Text to Image loss
            pos_sim = sim_t2i[i][pos_mask]
            neg_sim = sim_t2i[i][neg_mask]
            
            if len(pos_sim) > 0 and len(neg_sim) > 0:
                pos_loss = torch.relu(self.margin1 - pos_sim).mean()
                neg_loss = torch.relu(neg_sim - self.margin2 + self.margin1).mean()
                loss_t2i += pos_loss + self.alpha * neg_loss
        
        return (loss_i2t + loss_t2i) / batch_size

class AdaptiveHashCoding(nn.Module):
    def __init__(self, main_dim, aux_dim, num_aux_codes=10):
        super().__init__()
        self.aux_library = nn.Parameter(torch.randn(num_aux_codes, aux_dim))
        self.attention_net = nn.Sequential(
            nn.Linear(main_dim + aux_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, main_features, current_state=None):
        batch_size = main_features.size(0)
        expanded_main = main_features.unsqueeze(1).repeat(1, self.aux_library.size(0), 1)
        expanded_aux = self.aux_library.unsqueeze(0).repeat(batch_size, 1, 1)
        
        concat_features = torch.cat([expanded_main, expanded_aux], dim=-1)
        attention_weights = F.softmax(self.attention_net(concat_features).squeeze(-1), dim=1)
        
        weighted_aux = torch.sum(attention_weights.unsqueeze(-1) * expanded_aux, dim=1)
        
        return main_features + weighted_aux