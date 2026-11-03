import os
import sys
import shutil
import random
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataloaders import BaseDataSets, TwoStreamBatchSampler
from dataloaders.transform import WeakStrongAugment
from networks import net_factory
from utils import losses, val_2d, config
from utils import patients_to_slices, get_current_consistency_weight, update_ema_variables

# ===== MSCP-Net 模块导入 =====
from utils.mscpnet import (
    ScaleAwareReweight as PASRModule,
    view_variance_enhance_2d as cfca_augment,
    compute_sdm,
    sdm_consistency_loss as gabc_loss,
)


def train(args, snapshot_path):
    """MT-Clean + MSCP-Net (PASR / CFCA / GABC 三个独立模块)."""
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.batch_size % 2 == 0

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_channels=1, num_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    model.train()

    # ===== MSCP-Net 三个独立开关 (与论文命名对齐) =====
    use_pasr    = getattr(args, 'use_pasr', 0)
    use_cfca    = getattr(args, 'use_cfca', 0)
    use_gabc    = getattr(args, 'use_gabc', 0)
    cfca_alpha  = getattr(args, 'cfca_alpha', 0.1)
    beta_gabc   = getattr(args, 'beta_gabc', 0.1)
    T_sdm       = getattr(args, 'T_sdm', 2000)
    beta_pasr   = getattr(args, 'beta_pasr', 0.5)

    # PASR 模块 (只创建一次, 无论是否启用)
    pasr_module = PASRModule(num_classes).cuda()

    # 把 PASR 模块的参数加入优化器
    optimizer_params = list(model.parameters()) + list(pasr_module.parameters())

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([WeakStrongAugment(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    logging.info(f"Total slices: {total_slices}, labeled slices: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4,
        pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
    valloader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=1,
        persistent_workers=True, pin_memory=True)

    optimizer = optim.SGD(optimizer_params, lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss()
    dc_loss = losses.MaskDiceLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, train_batch in enumerate(trainloader):
            image_w = train_batch['image_w'].float().cuda()
            image_s = train_batch['image_s'].float().cuda()
            gt_w = train_batch['label_w'].long().cuda()

            image_wl, image_wu = image_w.chunk(2)
            _, image_su = image_s.chunk(2)
            gt_wl, _ = gt_w.chunk(2)

            # ============================================================
            # CFCA: 临床频域校准增强 (2D 傅里叶域)
            # ============================================================
            if use_cfca:
                image_wu_in = image_wu                    # teacher 用无扰动原图
                image_su_in = cfca_augment(image_su, alpha=cfca_alpha)  # student 用频域扰动图
            else:
                image_wu_in = image_wu
                image_su_in = image_su

            # ---------- 教师伪标签 ----------
            with torch.no_grad():
                logit_wu_ema, _ = ema_model(image_wu_in)
                prob_wu_mean = torch.softmax(logit_wu_ema, dim=1)
                pred_wu = prob_wu_mean.argmax(dim=1)   # [B_u, H, W]

            # ---------- 学生强增强预测 ----------
            logit_su, aux_su = model(image_su_in)
            prob_su = torch.softmax(logit_su, dim=1)
            loss_con = dc_loss(prob_su, pred_wu.unsqueeze(1))

            # ---------- 标注监督 (有监督损失) ----------
            logit_wl, _ = model(image_wl)
            prob_wl = logit_wl.softmax(dim=1)
            loss_sup = (ce_loss(logit_wl, gt_wl)
                      + dc_loss(prob_wl, gt_wl.unsqueeze(1))) / 2.0

            # ============================================================
            # PASR: 像素级尺度调和 (双分支)
            # ============================================================
            loss_pasr = torch.tensor(0.0, device=image_w.device)
            if use_pasr:
                low_pred  = aux_su['low_out']    # [B, C, h, w]
                high_pred = aux_su['high_out']   # [B, C, h, w]

                if low_pred.shape[-2:] != pred_wu.shape[-2:]:
                    low_pred = F.interpolate(
                        low_pred, size=pred_wu.shape[-2:],
                        mode='bilinear', align_corners=True)
                if high_pred.shape[-2:] != pred_wu.shape[-2:]:
                    high_pred = F.interpolate(
                        high_pred, size=pred_wu.shape[-2:],
                        mode='bilinear', align_corners=True)

                prob_low  = torch.softmax(low_pred,  dim=1)
                prob_high = torch.softmax(high_pred, dim=1)

                with torch.no_grad():
                    conf_low  = prob_low.max(dim=1).values
                    conf_high = prob_high.max(dim=1).values

                ensemble_pred, _, _ = pasr_module(
                    prob_low, prob_high, conf_low, conf_high)

                loss_pasr = dc_loss(ensemble_pred, pred_wu.unsqueeze(1))

            # ============================================================
            # GABC: 几何锚定边界一致性
            # ============================================================
            loss_gabc = torch.tensor(0.0, device=image_w.device)
            if use_gabc:
                sdm_pred = aux_su['sdm']         # [B, C, H, W]
                with torch.no_grad():
                    sdm_target = compute_sdm(pred_wu, num_classes)
                if sdm_pred.shape[-2:] != sdm_target.shape[-2:]:
                    sdm_pred = F.interpolate(
                        sdm_pred, size=sdm_target.shape[-2:],
                        mode='bilinear', align_corners=True)
                loss_gabc = gabc_loss(sdm_pred, sdm_target)

            # ---------- Total Loss (对齐论文 Eq. 2) ----------
            lambda_t = get_current_consistency_weight(iter_num // 150)
            s_t = min(1.0, float(iter_num) / float(max(T_sdm, 1)))

            loss = loss_sup + lambda_t * loss_con
            if use_pasr:
                loss = loss + beta_pasr * loss_pasr
            if use_gabc:
                loss = loss + s_t * beta_gabc * loss_gabc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # ---------- Logging ----------
            iter_num += 1
            writer.add_scalar('info/loss', loss, iter_num)
            writer.add_scalar('info/loss_sup', loss_sup, iter_num)
            writer.add_scalar('info/loss_con', loss_con, iter_num)
            writer.add_scalar('info/loss_pasr', loss_pasr, iter_num)
            writer.add_scalar('info/loss_gabc', loss_gabc, iter_num)
            writer.add_scalar('info/lambda_t', lambda_t, iter_num)
            logging.info(
                'iteration %d : loss : %.4f, loss_sup: %.4f, loss_con: %.4f, '
                'loss_pasr: %.4f, loss_gabc: %.4f' %
                (iter_num, loss, loss_sup, loss_con,
                 loss_pasr.item(), loss_gabc.item()))

            # ---------- Validation ----------
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for val_batch in tqdm(valloader, total=len(db_val),
                                      desc=f"Validation {iter_num}", unit="case"):
                    metric_i = val_2d.test_single_volume(
                        val_batch["image"], val_batch["label"], model,
                        classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_asd'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path, 'iter_{}_dice_{}.pth'.format(
                            iter_num, round(best_performance, 4)))
                    save_best = os.path.join(
                        snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                iterator.close()
                writer.close()
                return "Training Finished"


if __name__ == "__main__":
    args = config.get_config(model='unet_proj')

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 日志文件夹命名也同步更新为 pasr / cfca / gabc
    snapshot_path = (
        f"log/{args.name}/{args.exp}_{args.labelnum}_{args.model}"
        f"_pasr{getattr(args, 'use_pasr', 0)}"
        f"_cfca{getattr(args, 'use_cfca', 0)}"
        f"_gabc{getattr(args, 'use_gabc', 0)}_r{args.round}"
    )
    os.makedirs(snapshot_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(snapshot_path, os.path.basename(__file__)))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
