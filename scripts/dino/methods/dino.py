

import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, student_temp=0.2, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        

    def forward(self, student_output, teacher_output):
        """
        student_output: list of tensors [B, out_dim] from multiple views (ncrops)
        teacher_output: list of tensors [B, out_dim] from global views (typically 2)
        """
        student_out = [s / self.student_temp for s in student_output]
        student_log_probs = [F.log_softmax(s, dim=-1) for s in student_out]

        teacher_out = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                       for t in teacher_output]

        total_loss = 0.0
        n_loss_terms = 0

        for iq, tq in enumerate(teacher_out):  # only global crops
            for v, sp in enumerate(student_log_probs):
                if v == iq:
                    continue
                loss = -torch.sum(tq * sp, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # update center
        with torch.no_grad():
            batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))

        return total_loss
