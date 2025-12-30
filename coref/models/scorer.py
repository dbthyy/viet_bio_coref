import torch
import torch.nn as nn

class CorefScorer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*3 + 2, 512),  
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, mi, mj, dist, same_string):
        x = torch.cat(
            [mi, mj, torch.abs(mi - mj), dist, same_string],
            dim=-1
        )
        return self.mlp(x).squeeze(-1)
    
def compute_pairwise_scores(model, mention_embs):
    model.eval()
    device = mention_embs.device
    N = mention_embs.size(0)
    scores = torch.zeros(N, N, device=device)

    with torch.no_grad():
        for i in range(N):
            for j in range(i + 1, N):
                dist = torch.log(
                    torch.tensor([[j - i + 1]], device=device)
                )
                same_string = torch.zeros(1, 1, device=device)

                s = model(
                    mention_embs[i:i+1],
                    mention_embs[j:j+1],
                    dist,
                    same_string
                )
                scores[i, j] = torch.sigmoid(s)

    return scores