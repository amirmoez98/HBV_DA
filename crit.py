import torch
import numpy as np
import math


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior='gauss'):
        super(SigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == 'gauss':
                loss = torch.exp(-s).mul((p - t)**2) / 2 + s / 2
            elif self.prior[0] == 'invGamma':
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = torch.exp(-s).mul(
                    (p - t)**2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
            lossMean = lossMean + torch.mean(loss)
        return lossMean


class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

class RmseLossRunoff(torch.nn.Module):
    def __init__(self):
        super(RmseLossRunoff, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = torch.log10(torch.sqrt(output[:, :, k]) + 0.1)
            t0 = torch.log10(torch.sqrt(target[:, :, k]) + 0.1)
            # p0 = torch.sqrt(output[:, :, k])
            # t0 = torch.sqrt(target[:, :, k])
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss


import torch
import torch.nn as nn

import torch
import torch.nn as nn

class RmseLossComb8(nn.Module):
    def __init__(self, alpha, gamma=.01, beta=1e-6):
        super(RmseLossComb8, self).__init__()
        self.alpha = alpha  # Weight for log-sqrt RMSE (fixed at 0.25)
        self.gamma = gamma  # Weight for PBiaslow, ensuring it doesn't introduce negative loss
        self.beta = beta    # Small number to avoid log(0)

    def forward(self, output, target):
        ny = target.shape[2]
        total_loss = 0

        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(p0 + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(t0 + self.beta) + 0.1)

            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss_rmse = torch.sqrt(((p - t) ** 2).mean())

            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss_log_sqrt_rmse = torch.sqrt(((pa - ta) ** 2).mean())

            # Calculate PBiaslow with absolute to ensure non-negative
            num_elements = int(0.3 * t.numel())
            sorted_p, _ = torch.sort(p)
            sorted_t, _ = torch.sort(t)
            pbiaslow_loss = torch.abs(torch.sum(sorted_p[:num_elements] - sorted_t[:num_elements]) / torch.sum(torch.abs(sorted_t[:num_elements])))
            pbiaslow_loss *= 100  # Convert to percentage, ensuring non-negativity

            combined_loss = (1.0 - self.alpha) * loss_rmse + self.alpha * loss_log_sqrt_rmse
            combined_loss += self.gamma * pbiaslow_loss  # Add PBiaslow to the total loss

            total_loss += combined_loss

        return total_loss



import torch
import torch.nn as nn

class RmseLossComb10(nn.Module):
    def __init__(self, alpha, gamma=.01, beta=1e-6):
        super(RmseLossComb10, self).__init__()
        self.alpha = alpha  # Weight for log-sqrt RMSE
        self.gamma = gamma  # Weight for PBiaslow
        self.beta = beta    # Small number to avoid log(0)

    def forward(self, output, target):
        ny = target.shape[2]
        total_loss = 0

        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(p0 + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(t0 + self.beta) + 0.1)

            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss_rmse = torch.sqrt(((p - t) ** 2).mean())

            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss_log_sqrt_rmse = torch.sqrt(((pa - ta) ** 2).mean())

            # Modified PBias calculation
            num_elements = int(0.3 * t.numel())
            sorted_p, _ = torch.sort(p)
            sorted_t, _ = torch.sort(t)
            pbiaslow = (sorted_p[:num_elements] - sorted_t[:num_elements]) / torch.sum(torch.abs(sorted_t[:num_elements]))
            pbiaslow_loss = torch.sum(pbiaslow ** 2) * 100  # Squaring to ensure non-negativity

            combined_loss = (1.0 - self.alpha) * loss_rmse + self.alpha * loss_log_sqrt_rmse
            combined_loss += self.gamma * pbiaslow_loss

            total_loss += combined_loss

        return total_loss

class RmseLossComb9(torch.nn.Module):
    def __init__(self, alpha, gamma=.1, beta=1e-6):
        super(RmseLossComb9, self).__init__()
        self.alpha = alpha  # Weight for log-sqrt RMSE (fixed at 0.25)
        self.gamma = gamma  # Weight for PBiaslow
        self.beta = beta    # Small number to avoid log(0)

    def forward(self, output, target):
        ny = target.shape[2]
        total_loss = 0

        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(p0 + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(t0 + self.beta) + 0.1)

            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss_rmse = torch.sqrt(((p - t) ** 2).mean())

            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss_log_sqrt_rmse = torch.sqrt(((pa - ta) ** 2).mean())

            # Calculate PBiaslow
            num_elements = int(0.3 * t.numel())  # 30% of total elements
            sorted_p, _ = torch.sort(p)
            sorted_t, _ = torch.sort(t)
            pbiaslow_loss = torch.sum(sorted_p[:num_elements] - sorted_t[:num_elements]) / torch.sum(torch.abs(sorted_t[:num_elements]))
            pbiaslow_loss *= 100  # Convert to percentage

            combined_loss = (1.0 - self.alpha) * loss_rmse + self.alpha * loss_log_sqrt_rmse
            combined_loss += self.gamma * pbiaslow_loss  # Add PBiaslow to the total loss

            total_loss += combined_loss

        return total_loss






class RmseLossComb(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6):
        super(RmseLossComb, self).__init__()
        self.alpha = alpha # weights of log-sqrt RMSE
        self.beta = beta

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k]+self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k]+self.beta) + 0.1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean()) # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            # pa = torch.log10(torch.sqrt(p) + 0.1)
            # ta = torch.log10(torch.sqrt(t) + 0.1)
            loss2 = torch.sqrt(((pa - ta)**2).mean()) #Log-Sqrt RMSE item
            temp = (1.0-self.alpha)*loss1 + self.alpha*loss2
            loss = loss + temp
        return loss
import torch

class RmseLossComb3(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6, gamma=.01):
        super(RmseLossComb3, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.gamma = gamma  # weight of PBiaslow

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t) ** 2).mean())  # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss2 = torch.sqrt(((pa - ta) ** 2).mean())  # Log-Sqrt RMSE item

            # Calculate PBiaslow
            pred_sort = p.cpu().detach().numpy()
            target_sort = t.cpu().detach().numpy()
            indexlow = round(0.3 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            lowtarget = target_sort[:indexlow]
            PBiaslow = torch.tensor(
                (lowpred - lowtarget).sum() / lowtarget.sum() * 100, dtype=torch.float32
            )

            temp = (1.0 - self.alpha ) * loss1 + self.alpha * loss2 + self.gamma * PBiaslow
            loss = loss + temp
        return loss
import torch
import torch.nn as nn
import torch
import torch.nn as nn

class RmseLossComb5(nn.Module):
    def __init__(self, alpha, gamma=.4, delta=.3, beta=1e-6):
        super(RmseLossComb5, self).__init__()
        self.alpha = alpha  # Weight for log-sqrt RMSE
        self.gamma = gamma  # Weight for PBiaslow
        self.delta = delta  # Additional weight for RMSE on low flows
        self.beta = beta    # Small number to avoid log(0)

    def forward(self, output, target):
        ny = target.shape[2]
        total_loss = 0

        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(p0 + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(t0 + self.beta) + 0.1)

            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss_rmse = torch.sqrt(((p - t) ** 2).mean())

            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss_log_sqrt_rmse = torch.sqrt(((pa - ta) ** 2).mean())

            combined_loss = (1.0 - self.alpha) * loss_rmse + self.alpha * loss_log_sqrt_rmse

            # Ensure non-negative PBias
            num_elements = int(0.3 * t.numel())
            sorted_p, _ = torch.sort(p)
            sorted_t, _ = torch.sort(t)
            pbiaslow_loss = (torch.sum(sorted_p[:num_elements] - sorted_t[:num_elements]) / torch.sum(torch.abs(sorted_t[:num_elements]))) * 100

            # Low flow RMSE
            low_flow_rmse = torch.sqrt(((sorted_p[:num_elements] - sorted_t[:num_elements]) ** 2).mean())

            # Combine all losses and ensure it's non-negative
            total_loss += max(0, combined_loss + self.gamma * pbiaslow_loss + self.delta * low_flow_rmse)

        return total_loss

class RmseLossComb4(nn.Module):
    def __init__(self, alpha, gamma=.01, beta=1e-6):
        super(RmseLossComb4, self).__init__()
        self.alpha = alpha  # Weight for log-sqrt RMSE
        self.gamma = gamma  # Weight for PBiaslow
        self.beta = beta    # Small number to avoid log(0)

    def forward(self, output, target):
        ny = target.shape[2]
        total_loss = 0

        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(p0 + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(t0 + self.beta) + 0.1)

            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss_rmse = torch.sqrt(((p - t) ** 2).mean())

            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss_log_sqrt_rmse = torch.sqrt(((pa - ta) ** 2).mean())

            combined_loss = (1.0 - self.alpha) * loss_rmse + self.alpha * loss_log_sqrt_rmse

            # Calculate PBiaslow
            num_elements = int(0.3 * t.numel())  # 30% of the total number of elements
            sorted_p, _ = torch.sort(p)
            sorted_t, _ = torch.sort(t)
            pbiaslow_loss = torch.sum(sorted_p[:num_elements] - sorted_t[:num_elements]) / torch.sum(sorted_t[:num_elements])
            pbiaslow_loss *= 100  # Convert to percentage

            # Combine all losses
            total_loss += combined_loss + self.gamma * pbiaslow_loss

        return total_loss

class RmseLossComb2(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6, lambda_k=0.01):
        super(RmseLossComb2, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.lambda_k = lambda_k  # regularization strength

    def forward(self, output, target, k):
        ny = target.shape[2]
        loss = 0
        for i in range(ny):
            p0 = output[:, :, i]
            t0 = target[:, :, i]
            p1 = torch.log10(torch.sqrt(p0 + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(t0 + self.beta) + 0.1)

            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean())  # RMSE item

            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss2 = torch.sqrt(((pa - ta)**2).mean())  # Log-Sqrt RMSE item

            temp = (1.0 - self.alpha) * loss1 + self.alpha * loss2
            loss += temp

        # Add the regularization term for k
        reg_loss = self.lambda_k * (k ** 2).sum()  # L2 regularization on k
        total_loss = loss + reg_loss

        return total_loss


# Usage would be the same as the previous example


class RmseLossCNN(torch.nn.Module):
    def __init__(self):
        super(RmseLossCNN, self).__init__()

    def forward(self, output, target):
        # output = ngrid * nvar * ntime
        ny = target.shape[1]
        loss = 0
        for k in range(ny):
            p0 = output[:, k, :]
            t0 = target[:, k, :]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

class RmseLossANN(torch.nn.Module):
    def __init__(self, get_length=False):
        super(RmseLossANN, self).__init__()
        self.ind = get_length

    def forward(self, output, target):
        if len(output.shape) == 2:
            p0 = output[:, 0]
            t0 = target[:, 0]
        else:
            p0 = output[:, :, 0]
            t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        loss = torch.sqrt(((p - t)**2).mean())
        if self.ind is False:
            return loss
        else:
            Nday = p.shape[0]
            return loss, Nday

class ubRmseLoss(torch.nn.Module):
    def __init__(self):
        super(ubRmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            pmean = p.mean()
            tmean = t.mean()
            p_ub = p-pmean
            t_ub = t-tmean
            temp = torch.sqrt(((p_ub - t_ub)**2).mean())
            loss = loss + temp
        return loss

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = ((p - t)**2).mean()
            loss = loss + temp
        return loss

class NSELoss(torch.nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample +1
        # minimize the opposite average NSE
        loss = -(losssum/nsample)
        return loss

class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                # temp = SSRes / ((torch.sqrt(SST)+0.1)**2)
                temp = SSRes / (SST+0.1)
                losssum = losssum + temp
                nsample = nsample +1
        loss = losssum/nsample
        return loss

# class NSELossBatch1(torch.nn.Module):
#     # Same as Fredrick 2019, batch NSE loss
#     # stdarray: the standard deviation of the runoff for all basins
#     def __init__(self, stdarray, eps=0.1):
#         super(NSELossBatch, self).__init__()
#         self.std = stdarray
#         self.eps = eps
#
#     def forward(self, output, target, igrid):
#
#         # nt = target.shape[0]  # Number of time steps
#         #
#         # stdse = np.tile(self.std[igrid], (nt, 1))
#         # stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
#         # nt = target.shape[0]
#         # stdse = np.tile(self.std[igrid], (nt, 1))
#         # stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
#         nt = target.shape[0]  # Number of time steps
#         nb = target.shape[1]  # Number of basins
#
#         # Ensure std is repeated correctly across all time steps for each basin
#         stdse = np.tile(self.std, (nt, 1))
#         stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
#         p0 = output[:, :, 0]   # dim: Time*Gage
#         t0 = target[:, :, 0]
#         mask = t0 == t0
#         p = p0[mask]
#         t = t0[mask]
#         stdw = stdbatch[mask]
#         sqRes = (p - t)**2
#         normRes = sqRes / (stdw + self.eps)**2
#         loss = torch.mean(normRes)
#
#         # sqRes = (t0 - p0)**2 # squared error
#         # normRes = sqRes / (stdbatch + self.eps)**2
#         # mask = t0 == t0
#         # loss = torch.mean(normRes[mask])
#         return loss

class NSELossBatch(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray,  eps=0.1):
        super(NSELossBatch, self).__init__()
        self.std = stdarray
        self.eps = eps


    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid].T, (nt, 1))

        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
        p0 = output[:, :, 0]   # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss





class NSELossBatch2(torch.nn.Module):
    def __init__(self, stdarray, eps=0.1):
        super(NSELossBatch2, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, output, target):
        nt = target.shape[0]  # Number of time steps
        nb = target.shape[1]  # Number of basins

        # Tile the standard deviation for each basin across all time steps
        stdse = np.tile(self.std, (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()

        p0 = output[:, :, 0]   # predictions for first variable (e.g., precipitation)
        t0 = target[:, :, 0]   # targets for first variable
        mask = t0 == t0  # Mask to filter out NaN values

        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]  # Apply mask to standardized tensor

        sqRes = (p - t)**2  # Squared residuals
        normRes = sqRes / (stdw + self.eps)**2  # Normalized residuals
        loss = torch.mean(normRes)  # Calculate mean loss

        return loss




class NSELossBatch3(torch.nn.Module):
    def __init__(self, stdarray, eps=0.1):
        super(NSELossBatch3, self).__init__()
        self.std = stdarray  # This should be an array of standard deviations for each of the 531 basins
        self.eps = eps

    def forward(self, output, target):
        nt, nb, _ = target.shape  # Extract dimensions: time steps, number of basins, and features

        # Tile the standard deviation for each basin across all time steps
        stdse = np.tile(self.std.reshape(1, -1), (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()

        p0 = output[:, :, 0]  # predictions for first variable (e.g., precipitation)
        t0 = target[:, :, 0]  # targets for first variable
        target_mask = t0 == t0  # False where target is NaN
        pred_mask = p0 == p0    # False where prediction is NaN

        # Combine masks - only True where both prediction and target are valid
        valid_mask = target_mask & pred_mask

        # Apply mask to all relevant tensors
        p = p0[valid_mask]
        t = t0[valid_mask]
        stdw = stdbatch[valid_mask]
         #Create masks for both predictions and targets
        # mask = t0 == t0  # Mask to filter out NaN values

        # p = p0[mask]
        # t = t0[mask]
        #stdw = stdbatch[mask]  # Apply mask to standardized tensor, ensuring shape alignment

        sqRes = (p - t)**2  # Squared residuals
        normRes = sqRes / (stdw + self.eps)**2  # Normalized residuals
        loss = torch.mean(normRes)

        loss2 = normRes # Calculate mean loss

        return loss


class NSELossBatch33(torch.nn.Module):
    def __init__(self, stdarray, eps=0.1):
        super(NSELossBatch33, self).__init__()
        self.std = stdarray  # This should be an array of standard deviations for each of the 531 basins
        self.eps = eps

    def forward(self, output, target):
        nt, nb, _ = target.shape  # Extract dimensions: time steps, number of basins, and features

        # Tile the standard deviation for each basin across all time steps
        stdse = np.tile(self.std.reshape(1, -1), (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()

        # Initialize a tensor to hold the losses for each basin
        losses_per_basin = torch.zeros((nt, nb), device="cuda")

        # Calculate the normalized residuals for each time step and each basin
        for i in range(nb):
            p = output[:, i, 0]  # predictions for first variable for basin i
            t = target[:, i, 0]  # targets for first variable for basin i
            stdw = stdbatch[:, i]  # standard deviations for basin i

            mask = ~torch.isnan(t)  # create a mask for non-NaN target values
            p = p[mask]
            t = t[mask]
            stdw = stdw[mask]

            if mask.any():  # proceed only if there are non-NaN values
                sqRes = (p - t) ** 2  # squared residuals
                normRes = sqRes / (stdw + self.eps) ** 2  # normalized residuals
                losses_per_basin[mask, i] = normRes  # store per-basin losses for valid entries

        # Calculate the mean loss for each basin over all valid time steps
        loss_per_basin = torch.mean(losses_per_basin, dim=0)  # average over time steps
        mean_loss = torch.mean(loss_per_basin)  # overall mean loss

        return mean_loss, loss_per_basin


class NSELossBatch4(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray,  eps=0.1,device = 'cpu'):
        super(NSELossBatch4, self).__init__()
        self.std = stdarray
        self.eps = eps
        self.device = device

    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid].T, (nt, 1))

        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(self.device)
        p0 = output[:, :, 0]   # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss

class NSESqrtLossBatch(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss, use RMSE and STD instead
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, eps=0.1):
        super(NSESqrtLossBatch, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid], (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
        p0 = output[:, :, 0]   # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = torch.sqrt((p - t)**2)
        normRes = sqRes / (stdw + self.eps)
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss

class TrendLoss(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(TrendLoss, self).__init__()

    def getSlope(self, x):
        idx = 0
        n = len(x)
        d = torch.ones(int(n * (n - 1) / 2))

        for i in range(n - 1):
            j = torch.arange(start=i + 1, end=n)
            d[idx: idx + len(j)] = (x[j] - x[i]) / (j - i).type(torch.float)
            idx = idx + len(j)

        return torch.median(d)

    def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
        # output, target: rho/time * Batchsize * Ntraget_var
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        loss = 0
        for k in range(ny):
        # loop for variable
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            # first part loss, regular RMSE
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
            temptrendloss = 0
            nsample = 0
            for ig in range(ngage):
                # loop for basins
                pgage0 = p0[:, ig].reshape(-1, 365)
                tgage0 = t0[:, ig].reshape(-1, 365)
                gBool = np.zeros(tgage0.shape[0]).astype(int)
                pgageM = torch.zeros(tgage0.shape[0])
                pgageQ = torch.zeros(tgage0.shape[0], len(PercentLst))
                tgageM = torch.zeros(tgage0.shape[0])
                tgageQ = torch.zeros(tgage0.shape[0], len(PercentLst))
                for ii in range(tgage0.shape[0]):
                    pgage = pgage0[ii, :]
                    tgage = tgage0[ii, :]
                    maskg = tgage == tgage
                    # quality control
                    if maskg.sum() > (1-2/12)*365:
                        gBool[ii] = 1
                        pgage = pgage[maskg]
                        tgage = tgage[maskg]
                        pgageM[ii] = pgage.mean()
                        tgageM[ii] = tgage.mean()
                        for ip in range(len(PercentLst)):
                            k = math.ceil(PercentLst[ip] / 100 * 365)
                            # pgageQ[ii, ip] = torch.kthvalue(pgage, k)[0]
                            # tgageQ[ii, ip] = torch.kthvalue(tgage, k)[0]
                            pgageQ[ii, ip] = torch.sort(pgage)[0][k-1]
                            tgageQ[ii, ip] = torch.sort(tgage)[0][k-1]
                # Quality control
                if gBool.sum()>6:
                    nsample = nsample + 1
                    pgageM = pgageM[gBool]
                    tgageM = tgageM[gBool]
                    # mean annual trend loss
                    temptrendloss = temptrendloss + (self.getSlope(tgageM)-self.getSlope(pgageM))**2
                    pgageQ = pgageQ[gBool, :]
                    tgageQ = tgageQ[gBool, :]
                    # quantile trend loss
                    for ii in range(tgageQ.shape[1]):
                        temptrendloss = temptrendloss + (self.getSlope(tgageQ[:, ii])-self.getSlope(pgageQ[:, ii]))**2

            loss = loss + temptrendloss/nsample

        return loss


class ModifyTrend(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(ModifyTrend, self).__init__()

    def getSlope(self, x):
        nyear, ngage = x.shape
        # define difference matirx
        x = x.transpose(0,1)
        xtemp = x.repeat(1, nyear)
        xi = xtemp.reshape([ngage, nyear, nyear])
        xj = xi.transpose(1,2)
        # define i,j matrix
        im = torch.arange(nyear).repeat(nyear).reshape(nyear,nyear).type(torch.float)
        im = im.unsqueeze(0).repeat([ngage, 1, 1])
        jm = im.transpose(1,2)
        delta = 1.0/(im - jm)
        delta = delta.cuda()
        # calculate the slope matrix
        slopeMat = (xi - xj)*delta
        rid, cid = np.triu_indices(nyear, k=1)
        slope = slopeMat[:, rid, cid]
        senslope = torch.median(slope, dim=-1)[0]

        return senslope


    def forward(self, output, target, PercentLst=[-1]):
        # output, target: rho/time * Batchsize * Ntraget_var
        # PercentLst = [100, 98, 50, 30, 2, -1]
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        # loop for variable
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        # first part loss, regular RMSE
        # loss = torch.sqrt(((p - t)**2).mean())
        # loss = ((p - t) ** 2).mean()
        loss = 0
        temptrendloss = 0
        # second loss: adding trend
        p1 = p0.reshape(-1, 365, ngage)
        t1 = t0.reshape(-1, 365, ngage)
        for ip in range(len(PercentLst)):
            k = math.ceil(PercentLst[ip] / 100 * 365)
            # pQ = torch.kthvalue(p1, k, dim=1)[0]
            # tQ = torch.kthvalue(t1, k, dim=1)[0]
            # output: dim=Year*gage
            if PercentLst[ip]<0:
                pQ = torch.mean(p1, dim=1)
                tQ = torch.mean(t1, dim=1)
            else:
                pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
                tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
            # temptrendloss = temptrendloss + ((self.getSlope(pQ) - self.getSlope(tQ)) ** 2).mean()
            temptrendloss = temptrendloss + ((pQ - tQ) ** 2).mean()
        loss = loss + temptrendloss

        return loss


class ModifyTrend1(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(ModifyTrend1, self).__init__()

    def getM(self, n):
        M = np.zeros([n**2, n])
        s0 = np.zeros([n**2, 1])
        for j in range (n):
            for i in range(n):
                k = j*n+i
                if i<j:
                    factor = 1/(j-i)
                    M[k, j] = factor
                    M[k, i] = -factor
                else:
                    s0[k] = np.nan
        sind = np.argwhere(~np.isnan(s0))
        return M, sind


    def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
        # PercentLst = [100, 98, 50, 30, 2]
        # output, target: rho/time * Batchsize * Ntraget_var
        output = output.permute(2, 0, 1)
        target = target.permute(2, 0, 1)
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        # loop for variable
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]
        # mask = t0 == t0
        # p = p0[mask]
        # t = t0[mask]
        # first part loss, regular RMSE
        loss = 0.0
        # loss = torch.sqrt(((p - t)**2).mean())
        # loss = ((p - t) ** 2).mean()
        # second loss: adding trend
        p1 = p0.reshape(-1, 365, ngage)
        t1 = t0.reshape(-1, 365, ngage)
        nyear = p1.shape[0]
        nsample = p1.shape[-1]
        M, s0 = self.getM(nyear)
        Mtensor = torch.from_numpy(M).type(torch.float).cuda()
        for ip in range(len(PercentLst)):
            k = math.ceil(PercentLst[ip] / 100 * 365)
            # pQ = torch.kthvalue(p1, k, dim=1)[0]
            # tQ = torch.kthvalue(t1, k, dim=1)[0]
            # output: dim=Year*gage
            pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
            tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
            # pQ = p1[:, 100, :]
            # tQ = t1[:, 100, :]
            temptrenddiff = 0.0
            for ig in range(nsample):
                trenddiff = (torch.median(torch.mv(Mtensor, pQ[:, ig])[s0[:,0]]) - \
                            torch.median(torch.mv(Mtensor, tQ[:, ig])[s0[:,0]]))**2
                temptrenddiff = temptrenddiff + trenddiff
            temptrendloss = temptrenddiff/nsample
            loss = loss + temptrendloss
        return loss

# class ModifyTrend1(torch.nn.Module):
#     # Test MSE loss for each percentile
#     def __init__(self):
#         super(ModifyTrend1, self).__init__()
#
#     def getM(self, n):
#         M = np.zeros([n**2, n])
#         s0 = np.zeros([n**2, 1])
#         for j in range (n):
#             for i in range(n):
#                 k = j*n+i
#                 if i<j:
#                     factor = 1/(j-i)
#                     M[k, j] = factor
#                     M[k, i] = -factor
#                 else:
#                     s0[k] = np.nan
#         sind = np.argwhere(~np.isnan(s0))
#         return M, sind
#
#
#     def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
#         # PercentLst = [100, 98, 50, 30, 2]
#         # output, target: rho/time * Batchsize * Ntraget_var
#         output = output.permute(2, 0, 1)
#         target = target.permute(2, 0, 1)
#         ny = target.shape[2]
#         nt = target.shape[0]
#         ngage = target.shape[1]
#         # loop for variable
#         p0 = output[:, :, 0]
#         t0 = target[:, :, 0]
#         mask = t0 == t0
#         p = p0[mask]
#         t = t0[mask]
#         # first part loss, regular RMSE
#         # loss = 0.0
#         # loss = torch.sqrt(((p - t)**2).mean())
#         loss = ((p - t) ** 2).mean()
#         # second loss: adding trend
#         p1 = p0.reshape(-1, 365, ngage)
#         t1 = t0.reshape(-1, 365, ngage)
#         nyear = p1.shape[0]
#         nsample = p1.shape[-1]
#         M, s0 = self.getM(nyear)
#         Mtensor = torch.from_numpy(M).type(torch.float).cuda()
#         for ip in range(len(PercentLst)):
#             k = math.ceil(PercentLst[ip] / 100 * 365)
#             # output: dim=Year*gage
#             pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
#             tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
#             # calculate mse of each percentile flow
#             mask = tQ == tQ
#             ptemp = pQ[mask]
#             ttemp = tQ[mask]
#             temploss = ((ptemp - ttemp) ** 2).mean()
#             loss = loss + temploss
#         return loss
import torch

class RmseLossComb_new(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6, gamma=1.5):
        super(RmseLossComb_new, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.gamma = gamma  # weights of NSE

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k]+self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k]+self.beta) + 0.1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean())  # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss2 = torch.sqrt(((pa - ta)**2).mean())  # Log-Sqrt RMSE item

            # NSE Loss
            t_mean = t.mean()
            numerator = torch.sum((p - t)**2)
            denominator = torch.sum((t - t_mean)**2)
            nse = 1 - numerator / (denominator + self.beta)
            nse_loss = 1 - nse

            # Combine losses
            temp = (1.0-self.alpha)*loss1+ self.alpha*loss2 + self.gamma*nse_loss
            loss = loss + temp

        return loss


import torch

class RmseLossComb_new2(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6, gamma=.1, delta=0.0007):
        super(RmseLossComb_new2, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.gamma = gamma  # weights of NSE
        self.delta = delta  # weights of absFLV

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean())  # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss2 = torch.sqrt(((pa - ta)**2).mean())  # Log-Sqrt RMSE item

            # NSE Loss
            t_mean = t.mean()
            numerator = torch.sum((p - t)**2)
            denominator = torch.sum((t - t_mean)**2)
            nse = 1 - numerator / (denominator + self.beta)
            nse_loss = 1 - nse

            # absFLV for the bottom 30% in log space
            sorted_indices = torch.argsort(t)
            indexlow = round(0.3 * len(sorted_indices))
            low_flow_indices = sorted_indices[:indexlow]
            lowpred = torch.log10(p[low_flow_indices] + self.beta)
            lowtarget = torch.log10(t[low_flow_indices] + self.beta)
            absFLV = torch.sum(torch.abs(lowpred - lowtarget)) / (torch.sum(lowtarget) + 0.0001) * 100

            # Combine losses
            temp = (1.0 - self.alpha) * loss1 + self.alpha * loss2 +  self.delta * absFLV
            loss = loss + temp

        return loss


import torch

class RmseLossComb_new3(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6, gamma=0.5, delta=0.001, epsilon=0.1):
        super(RmseLossComb_new3, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.gamma = gamma  # weights of NSE
        self.delta = delta  # weights of absFLV
        self.epsilon = epsilon  # weight to balance NSE and FLV connection

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean())  # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss2 = torch.sqrt(((pa - ta)**2).mean())  # Log-Sqrt RMSE item

            # NSE Loss
            t_mean = t.mean()
            numerator = torch.sum((p - t)**2)
            denominator = torch.sum((t - t_mean)**2)
            nse = 1 - numerator / (denominator + self.beta)
            nse_loss = 1 - nse

            # absFLV for the bottom 30% in log space
            sorted_indices = torch.argsort(t)
            indexlow = round(0.3 * len(sorted_indices))
            low_flow_indices = sorted_indices[:indexlow]
            lowpred = torch.log10(p[low_flow_indices] + self.beta)
            lowtarget = torch.log10(t[low_flow_indices] + self.beta)
            absFLV = torch.sum(torch.abs(lowpred - lowtarget)) / (torch.sum(lowtarget) + self.beta) * 100

            # Connection between NSE and FLV
            combined_nse_flv = self.epsilon * nse_loss + (1 - self.epsilon) * absFLV

            # Combine losses
            temp = (1.0 - self.alpha) * loss1 + self.alpha * loss2 + self.gamma * combined_nse_flv
            loss = loss + temp

        return loss



import torch

class RmseLossComb_new9(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6, gamma=1.5, dynamic_weight=True):
        super(RmseLossComb_new9, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.gamma = gamma  # weights of NSE
        self.dynamic_weight = dynamic_weight

    def forward(self, output, target, epoch=None):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean())  # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss2 = torch.sqrt(((pa - ta)**2).mean())  # Log-Sqrt RMSE item

            # NSE Loss
            t_mean = t.mean()
            numerator = torch.sum((p - t)**2)
            denominator = torch.sum((t - t_mean)**2)
            nse = 1 - numerator / (denominator + self.beta)
            nse_loss = 1 - nse

            # Adjust weights dynamically if specified
            if self.dynamic_weight and epoch is not None:
                dynamic_gamma = self.gamma * (1 + 2 * epoch)  # Increase gamma over epochs
            else:
                dynamic_gamma = self.gamma

            # Combine losses
            temp = (1.0 - self.alpha) * loss1 + self.alpha * loss2 + dynamic_gamma * nse_loss
            loss = loss + temp

        return loss



import torch
import numpy as np

class CombinedLoss(torch.nn.Module):
    def __init__(self, stdarray, alpha, eps=0.1, beta=1e-6):
        super(CombinedLoss, self).__init__()
        self.std = stdarray  # Standard deviations for each basin
        self.alpha = alpha   # Weighting factor between NSE and RMSE
        self.eps = eps       # Small epsilon for numerical stability in NSE
        self.beta = beta     # Small beta for numerical stability in RMSE

    def forward(self, output, target):
        nt, nb, _ = target.shape

        # Prepare standardized deviations for NSE loss
        stdse = np.tile(self.std.reshape(1, -1), (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()

        nse_loss = 0
        rmse_loss = 0
        ny = target.shape[2]

        for k in range(ny):
            # NSE loss computation
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            stdw = stdbatch[mask]
            sqRes = (p - t) ** 2
            normRes = sqRes / (stdw + self.eps) ** 2
            nse_loss += torch.mean(normRes)

            # RMSE and Log-Sqrt RMSE loss computation
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss1 = torch.sqrt(((p - t) ** 2).mean())  # RMSE item
            loss2 = torch.sqrt(((pa - ta) ** 2).mean())  # Log-Sqrt RMSE item
            rmse_loss += (1.0 - self.alpha) * loss1 + self.alpha * loss2

        # Combining NSE and RMSE losses, equally weighted for simplicity
        total_loss = 1.2 * nse_loss + .2 * rmse_loss
        return total_loss
class CombinedLoss2(torch.nn.Module):
    def __init__(self, stdarray, alpha, eps=0.1, beta=1e-6):
        super(CombinedLoss2, self).__init__()
        self.std = stdarray
        self.alpha = alpha
        self.eps = eps
        self.beta = beta
        self.nse_weight = 0.85
        self.rmse_weight = 0.15

    def forward(self, output, target):
        nt, nb, _ = target.shape
        stdse = np.tile(self.std.reshape(1, -1), (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()

        nse_loss = 0
        rmse_loss = 0
        ny = target.shape[2]

        for k in range(ny):
            # NSE loss computation
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            stdw = stdbatch[mask]
            sqRes = (p - t) ** 2
            normRes = sqRes / (stdw + self.eps) ** 2
            nse_loss += torch.mean(normRes)

            # RMSE and Log-Sqrt RMSE loss computation
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss1 = torch.sqrt(((p - t) ** 2).mean())
            loss2 = torch.sqrt(((pa - ta) ** 2).mean())
            rmse_loss += (1.0 - self.alpha) * loss1 + self.alpha * loss2

        total_loss = self.nse_weight * nse_loss + self.rmse_weight * rmse_loss
        return total_loss
import torch
import numpy as np

class CombinedLoss3(torch.nn.Module):
    def __init__(self, stdarray, alpha, eps=0.1, beta=1e-6):
        super(CombinedLoss3, self).__init__()
        self.std = stdarray  # Standard deviations for each basin
        self.alpha = alpha   # Weighting factor between NSE and RMSE
        self.eps = eps       # Small epsilon for numerical stability in NSE
        self.beta = beta     # Small beta for numerical stability in RMSE

    def forward(self, output, target):
        nt, nb, _ = target.shape
        stdse = np.tile(self.std.reshape(1, -1), (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()

        nse_loss = 0
        rmse_loss = 0
        kge_loss = 0
        ny = target.shape[2]

        for k in range(ny):
            # NSE loss computation
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            stdw = stdbatch[mask]
            sqRes = (p - t) ** 2
            normRes = sqRes / (stdw + self.eps) ** 2
            nse_loss += torch.mean(normRes)

            # RMSE and Log-Sqrt RMSE loss computation
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss1 = torch.sqrt(((p - t) ** 2).mean())
            loss2 = torch.sqrt(((pa - ta) ** 2).mean())
            rmse_loss += (1.0 - self.alpha) * loss1 + self.alpha * loss2

            # KGE computation
            r = torch.corrcoef(torch.stack((p, t)))[0, 1]
            alpha_kge = torch.std(p) / torch.std(t)
            beta_kge = torch.mean(p) / torch.mean(t)
            kge_loss += 1 - torch.sqrt((r-1)**2 + (alpha_kge-1)**2 + (beta_kge-1)**2)

        # Combining NSE, RMSE, and KGE losses
        total_loss = nse_loss + 0.5 * rmse_loss + 0.2 * kge_loss
        return total_loss
import torch

class CombinedLoss4(torch.nn.Module):
    def __init__(self, stdarray, alpha, eps=0.1, beta=1e-6):
        super(CombinedLoss4, self).__init__()
        self.std = stdarray  # Standard deviations for each basin
        self.alpha = alpha   # Weighting factor between NSE and RMSE
        self.eps = eps       # Small epsilon for numerical stability in NSE
        self.beta = beta     # Small beta for numerical stability in RMSE

    def forward(self, output, target):
        nt, nb, _ = target.shape
        stdse = np.tile(self.std.reshape(1, -1), (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()

        nse_loss = 0
        rmse_loss = 0
        kge_loss = 0
        ny = target.shape[2]

        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            stdw = stdbatch[mask]
            sqRes = (p - t) ** 2
            normRes = sqRes / (stdw + self.eps) ** 2
            nse_loss += torch.mean(normRes)

            # RMSE and Log-Sqrt RMSE loss computation
            p1 = torch.log10(torch.sqrt(output[:, :, k] + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k] + self.beta) + 0.1)
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            loss1 = torch.sqrt(((p - t) ** 2).mean())
            loss2 = torch.sqrt(((pa - ta) ** 2).mean())
            rmse_loss += (1.0 - self.alpha) * loss1 + self.alpha * loss2

            # Manual computation of Pearson correlation coefficient
            p_mean = torch.mean(p)
            t_mean = torch.mean(t)
            p_centered = p - p_mean
            t_centered = t - t_mean
            r_num = torch.sum(p_centered * t_centered)
            r_den = torch.sqrt(torch.sum(p_centered ** 2) * torch.sum(t_centered ** 2))
            r = r_num / r_den if r_den != 0 else 0

            alpha_kge = torch.std(p) / torch.std(t)
            beta_kge = torch.mean(p) / torch.mean(t)
            kge_loss += 1 - torch.sqrt((r-1)**2 + (alpha_kge-1)**2 + (beta_kge-1)**2)

        # Combining NSE, RMSE, and KGE losses
        total_loss = 1.05*nse_loss + 1 * rmse_loss + .5* kge_loss
        return total_loss
