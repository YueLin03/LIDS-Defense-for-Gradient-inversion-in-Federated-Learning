import lpips
import torch
import torchvision
import warnings

def cw_ssim(img_batch, ref_batch, scales=5, skip_scales=None, K=1e-6, reduction="mean"):
    """Batched complex wavelet structural similarity.

    As in Zhou Wang and Eero P. Simoncelli, "TRANSLATION INSENSITIVE IMAGE SIMILARITY IN COMPLEX WAVELET DOMAIN"
    Ok, not quite, this implementation computes no local SSIM and neither averaging over local patches and uses only
    the existing wavelet structure to provide a similar scale-invariant decomposition.

    skip_scales can be a list like [True, False, False, False] marking levels to be skipped.
    K is a small fudge factor.
    """
    try:
        from pytorch_wavelets import DTCWTForward
    except ModuleNotFoundError:
        warnings.warn(
            "To utilize wavelet SSIM, install pytorch wavelets from https://github.com/fbcotter/pytorch_wavelets."
        )
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    # 1) Compute wavelets:
    setup = dict(device=img_batch.device, dtype=img_batch.dtype)
    if skip_scales is not None:
        include_scale = [~s for s in skip_scales]
        total_scales = scales - sum(skip_scales)
    else:
        include_scale = True
        total_scales = scales
    xfm = DTCWTForward(J=scales, biort="near_sym_b", qshift="qshift_b", include_scale=include_scale).to(**setup)
    img_coefficients = xfm(img_batch)
    ref_coefficients = xfm(ref_batch)

    # 2) Multiscale complex SSIM:
    ssim = 0
    for xs, ys in zip(img_coefficients[1], ref_coefficients[1]):
        if len(xs) > 0:
            xc = torch.view_as_complex(xs)
            yc = torch.view_as_complex(ys)

            conj_product = (xc * yc.conj()).sum(dim=2).abs()
            square_img = (xc * xc.conj()).abs().sum(dim=2)
            square_ref = (yc * yc.conj()).abs().sum(dim=2)

            ssim_val = (2 * conj_product + K) / (square_img + square_ref + K)
            ssim += ssim_val.mean(dim=[1, 2, 3])
    ssim = ssim / total_scales
    return ssim.mean().item(), ssim.max().item()


def compute_batch_order(lpips_scorer, rec_denormalized, ground_truth_denormalized, device):
    """Re-order a batch of images according to LPIPS statistics of source batch, trying to match similar images.

    This implementation basically follows the LPIPS.forward method, but for an entire batch."""
    from scipy.optimize import linear_sum_assignment  # Again a lazy import

    B_rec = rec_denormalized.shape[0]
    L = lpips_scorer.L
    B_gt = ground_truth_denormalized.shape[0]

    with torch.inference_mode():
        # Compute all features [assume sufficient memory is a given]
        features_rec = []
        for input in rec_denormalized:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_rec.append(layer_features)

        features_gt = []
        for input in ground_truth_denormalized:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_gt.append(layer_features)

        # Compute overall similarities:
        similarity_matrix = torch.zeros(B_gt, B_rec, device=device)
        for idx, x in enumerate(features_gt):
            for idy, y in enumerate(features_rec):
                for kk in range(L):
                    diff = (x[kk] - y[kk]) ** 2
                    similarity_matrix[idx, idy] += spatial_average(lpips_scorer.lins[kk](diff)).squeeze()
    try:
        gt_assignment, rec_assignment = linear_sum_assignment(similarity_matrix.cpu().numpy(), maximize=False)
    except ValueError:
        print(f"ValueError from similarity matrix {similarity_matrix.cpu().numpy()}")
        print("Returning trivial order...")
        rec_assignment = list(range(B))
    return torch.as_tensor(gt_assignment, device=device, dtype=torch.long), torch.as_tensor(rec_assignment, device=device, dtype=torch.long)


def normalize_tensor(in_feat, eps=1e-10):
    """From https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/__init__.py."""
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    """https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py ."""
    return in_tens.mean([2, 3], keepdim=keepdim)


def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0, clip=False):
    """Standard PSNR."""
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch) ** 2).reshape(B, -1).mean(dim=1)
        if any(mse_per_example == 0):
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
        elif not all(torch.isfinite(mse_per_example)):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            psnr_per_example = 10 * torch.log10(factor ** 2 / mse_per_example)
            return psnr_per_example.mean().item(), psnr_per_example.max().item()


device = "cuda:1"
order_batch = True
resize_transform = torchvision.transforms.Resize(224)

lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(device)


def run_metrics(rec_denormalized, ground_truth_denormalized, order_batch=True, log=False):
    """PSNR and LPIPS calculator

    Args:
        rec_denormalized (tensor): reocntruction image, shape should be (1, C, H, W)
        ground_truth_denormalized (tensor): target images, shape should be (B, C, H, W)
        order_batch (bool, optional): select images by lpips score. Defaults to True.
        log (bool, optional): print metrics. Defaults to False.

    Returns:
        dict: metrics
    """
    
    rec_denormalized = torch.clamp(rec_denormalized, 0, 1).to(device)
    ground_truth_denormalized = torch.clamp(ground_truth_denormalized, 0, 1).to(device)

    # left_dim = rec_denormalized.dim() - 2 
    # rec_denormalized = rec_denormalized.expand( -1, 3, *[-1]*left_dim)
    # ground_truth_denormalized = ground_truth_denormalized.expand( -1, 3, *[-1]*left_dim)

    if rec_denormalized.shape[2] != 224:
        rec_denormalized = resize_transform( rec_denormalized )
        ground_truth_denormalized = resize_transform( ground_truth_denormalized )


    if order_batch:
        selector, order = compute_batch_order(lpips_scorer, rec_denormalized, ground_truth_denormalized, device)
        rec_denormalized = rec_denormalized[order]
        ground_truth_denormalized = ground_truth_denormalized[selector]
    else:
        order = None
        selector = None

    mse_score = (rec_denormalized - ground_truth_denormalized).pow(2).mean(dim=[1, 2, 3])
    avg_mse, max_mse = mse_score.mean().item(), mse_score.max().item()
    avg_psnr, max_psnr = psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1)
    avg_ssim, max_ssim = cw_ssim(rec_denormalized, ground_truth_denormalized, scales=5)

    lpips_score = lpips_scorer(rec_denormalized, ground_truth_denormalized, normalize=True)
    avg_lpips, max_lpips = lpips_score.mean().item(), lpips_score.max().item()

    if log:
        print(f"Average MSE: {avg_mse:.4f}, Avg PSNR: {avg_psnr:.4f}, Avg LPIPS: {avg_lpips:.4f}")
    
    return {"avg_mse": avg_mse, "max_mse": max_mse, "avg_psnr": avg_psnr, "max_psnr": max_psnr, "avg_lpips": avg_lpips, "max_lpips": max_lpips, "avg_ssim": avg_ssim, "max_ssim": max_ssim, "order": order.detach().cpu(), "selector": selector.detach().cpu()}
