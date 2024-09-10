# https://github.com/deeplearning-wisc/scone/blob/119485bfcbf7b2fd7093e2070751348d9df330cb/CIFAR/train.py#L591

"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss.crossentropy import cross_entropy
from ..utils import apply_reduction, is_known, is_unknown, evaluate_energy_logistic_loss


def _energy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


class EnergyMarginLoss(nn.Module):
    """
    Augments the cross-entropy by a regularization term
    that aims to increase the energy gap between IN and OOD samples.
    This term is defined as

    .. math::
       \\mathcal{L}(x, y) = \\alpha
       \\Biggl \\lbrace
       {
      \\max(0, E(x)  - m_{in})^2 \\quad \\quad  \\quad  \\quad \\quad \\quad   \\text{if } y \\geq 0
        \\atop
       \\max(0, m_{out} - E(x))^2 \\quad \\quad \\quad  \\quad \\quad  \\text{ otherwise }
       }


    where :math:`E(x) = - \\log(\\sum_i e^{f_i(x)} )` is the energy of :math:`x`.

    :see Paper:
        `NeurIPS <https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf>`__

    :see Implementation: `GitHub <https://github.com/wetliu/energy_ood>`__
    """

    def __init__(
        self,
        full_train_loss,
        eta=1.00,
        false_alarm_cutoff=0.05,
        in_constraint_weight=1.00,
        ce_tol=2.00,
        ce_constraint_weight=1.00,
        out_constraint_weight=1.00,
        lr_lam=1.00,
        penalty_mult=1.50,
        constraint_tol=0.00
    ):
        """
        :param alpha: weighting parameter
        """
        super(EnergyMarginLoss, self).__init__()
        self.full_train_loss = torch.tensor(full_train_loss).float()
        self.eta = torch.tensor(eta).float() # might be buggy here
        self.false_alarm_cutoff = torch.tensor(false_alarm_cutoff).float()
        self.in_constraint_weight = torch.tensor(in_constraint_weight).float()
        self.lam = torch.tensor(0).float()
        self.lam2 = torch.tensor(0).float()
        self.ce_tol = torch.tensor(ce_tol).float()
        self.ce_constraint_weight = torch.tensor(ce_constraint_weight).float()
        self.out_constraint_weight = torch.tensor(out_constraint_weight).float()
        self.lr_lam = torch.tensor(lr_lam).float()
        self.penalty_mult = torch.tensor(penalty_mult).float()
        self.constraint_tol = torch.tensor(constraint_tol).float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, logistic_regression: torch.Tensor) -> torch.Tensor:
        """
        Calculates weighted sum of cross-entropy and the energy regularization term.

        :param logits: logits
        :param targets: labels
        """
        # for classification
        if len(logits.shape) == 2:
            energy_loss_in, energy_loss_out = self._sigmoid_loss(logits=logits, y=targets,logistic_regression=logistic_regression)
            loss_in = self._alm_in_distribution_constraint(energy_loss_in=energy_loss_in)
            loss_ce = F.cross_entropy(logits[is_known(targets)], targets[is_known(targets)])
            loss_ce = self._alm_cross_entropy_constraint(loss_ce=loss_ce)
        else:
            raise ValueError(f"Unsupported input shape: {logits.shape}")
        return apply_reduction(loss_ce + self.out_constraint_weight * energy_loss_out + loss_in, reduction=None)

    def _sigmoid_loss(self, logits: torch.Tensor, y: torch.Tensor, logistic_regression: torch.Tensor) -> torch.Tensor:
        # for classification
        known = is_known(y)
        
        energy_loss_in = torch.mean(torch.sigmoid(logistic_regression(
            (torch.logsumexp(logits[is_known(y)], dim=1)).unsqueeze(1)).squeeze()))
        energy_loss_out = torch.mean(torch.sigmoid(-logistic_regression(
            (torch.logsumexp(logits[is_unknown(y)], dim=1) - self.eta).unsqueeze(1)).squeeze()))
        return energy_loss_in, energy_loss_out

    def _alm_in_distribution_constraint(self, energy_loss_in: torch.Tensor) -> torch.Tensor:
        # for classification
        in_constraint_term = energy_loss_in - self.false_alarm_cutoff
        
        if self.in_constraint_weight * in_constraint_term + self.lam >= 0:
            in_loss = in_constraint_term * self.lam + self.in_constraint_weight / 2 * torch.pow(in_constraint_term, 2)
        else:
            in_loss = - torch.pow(self.lam, 2) * 0.5 / self.in_constraint_weight
        return in_loss

    def _alm_cross_entropy_constraint(self, loss_ce: torch.Tensor) -> torch.Tensor:
        # for classification
        loss_ce_constraint = loss_ce - self.ce_tol * self.full_train_loss
        
        if self.ce_constraint_weight * loss_ce_constraint + self.lam2 >= 0:
            loss_ce = loss_ce_constraint * self.lam2 + self.ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
        else:
            loss_ce = - torch.pow(self.lam2, 2) * 0.5 / self.ce_constraint_weight
        return loss_ce
    
    def update_hyperparameters(self, model, train_loader_in, logistic_regression):
        avg_sigmoid_energy_losses, _, avg_ce_loss = evaluate_energy_logistic_loss(model, train_loader_in, logistic_regression)
        
        # update lam
        in_term_constraint = avg_sigmoid_energy_losses -  self.false_alarm_cutoff
        if in_term_constraint * self.in_constraint_weight + self.lam >= 0:
            self.lam += self.lr_lam * in_term_constraint
        else:
            self.lam += -self.lr_lam * self.lam / self.in_constraint_weight
            
        # update lam2
        ce_constraint = avg_ce_loss - self.ce_tol * self.full_train_loss
        if ce_constraint * self.ce_constraint_weight + self.lam2 >= 0:
            self.lam2 += self.lr_lam * ce_constraint
        else:
            self.lam2 += -self.lr_lam * self.lam2 / self.ce_constraint_weight
        
        # update in-distribution weight for alm
        if in_term_constraint > self.constraint_tol:
            self.in_constraint_weight *= self.penalty_mult
            
        if ce_constraint > self.constraint_tol:
            self.ce_constraint_weight *= self.penalty_mult