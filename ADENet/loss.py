import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-6
class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(256, 2)
		
	def forward(self, x, labels=None):	
		x = x.squeeze(1)
		x = self.FC(x)
		if labels == None:
			predScore = x[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			nloss = self.criterion(x, labels)
			predScore = F.softmax(x, dim = -1)
			predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum

class lossA(nn.Module):
	def __init__(self):
		super(lossA, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)	
		nloss = self.criterion(x, labels)
		return nloss

class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()

		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)
		nloss = self.criterion(x, labels)
		return nloss


class loss_aud(nn.Module):
    def __init__(self):
        super(loss_aud, self).__init__()

    def sisnr(self, x, s, asd_pred=None, asd_gt=None, eps=1e-8):
        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))

        if asd_pred is not None and asd_gt is not None:
	    
            speech_gap_e_frame = x.size(0) * x.size(1) // asd_gt.size(0)
            frame_gap = asd_gt.size(0) // x.size(0)

            asd_p = torch.zeros(x.size()).to(x.device)
            asd_g = torch.zeros(x.size()).to(x.device)
            k = 0
            for n in range(x.size(0)):
                for i in range(0, x.size(1), speech_gap_e_frame):
                    asd_p[n][i:i + speech_gap_e_frame] = asd_pred[k]
                    asd_g[n][i:i + speech_gap_e_frame] = asd_gt[k]
                    k = n * frame_gap + i  // speech_gap_e_frame + 1

            # x = x + torch.mul(x, (asd_p + asd_g))
            x = x + torch.mul(x, asd_p )

        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        # S_target
        t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (
            l2norm(s_zm, keepdim=True)**2 + eps)

        # e_noise = pred - S_target
        # SI-SNR
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def cal_SISNR(source, estimate_source):
        """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
        Args:
            source: torch tensor, [batch size, sequence length]
            estimate_source: torch tensor, [batch size, sequence length]
        Returns:
            SISNR, [batch size]
        """
        assert source.size() == estimate_source.size()

        # Step 1. Zero-mean norm
        source = source - torch.mean(source, axis = -1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

        # Step 2. SI-SNR
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
        proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
        # e_noise = s' - s_target
        noise = estimate_source - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
        sisnr = 10 * torch.log10(ratio + EPS)

        return sisnr

    def forward(self, pre, label, asd_pred=None, asd_gt=None):
        # loss = 0
        # for i in range(pre.shape[0]):
        # 	loss += self.sisnr(pre[i], label[i])
        return -torch.mean(self.sisnr(pre, label, asd_pred, asd_gt))


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)
        return err
    
class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))

class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))