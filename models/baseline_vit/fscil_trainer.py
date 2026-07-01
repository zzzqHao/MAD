import datetime
import os
import os.path as osp
import random
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import dataloader.data_utils as data_utils
import utils
from scheduler.lr_scheduler import LinearWarmupCosineAnnealingLR
from .baseline_net import BaselineViTProtoNet
from .orco import SupConLoss, normalize, perturb_targets_norm_count, simplex_loss


def save_args_to_yaml(args, output_file):
    args_dict = OrderedDict(vars(args))
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as yaml_file:
        yaml.dump(args_dict, yaml_file)


def evaluate(model, testloader, args, session):
    test_class = args.base_class + session * args.way
    model.eval()
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for data, labels in tqdm(testloader, desc=f"[Eval][Session {session}]", dynamic_ncols=True, leave=False):
            data = data.cuda()
            labels = labels.cuda()
            logits, _ = model(data)
            all_targets.append(labels)
            all_logits.append(logits[:, :test_class])

    all_targets = torch.cat(all_targets)
    all_logits = torch.cat(all_logits, dim=0)
    preds = torch.argmax(all_logits, dim=1)
    top1 = (preds == all_targets).float().mean().item()

    base_acc = utils.Averager()
    novel_acc = utils.Averager()
    for class_id in all_targets.unique():
        mask = all_targets == class_id
        class_acc = (preds[mask] == all_targets[mask]).float().mean().item()
        if class_id < args.base_class:
            base_acc.add(class_acc)
        else:
            novel_acc.add(class_acc)

    return top1, novel_acc.item(), base_acc.item()


class FSCILTrainer:
    def __init__(self, args):
        self.args = data_utils.set_up_datasets(args)
        self.init_trlog()
        self.set_save_path()
        self.print_config()
        self.model = self.build_model()
        self.best_model_dict = deepcopy(self.model.state_dict())
        self.session_results = []
        self.epoch_logs = []

    def init_trlog(self):
        self.trlog = {
            "max_acc": [0.0] * self.args.sessions,
            "max_novel_acc": [0.0] * self.args.sessions,
            "max_base_acc": [0.0] * self.args.sessions,
            "max_hm": [0.0] * self.args.sessions,
        }

    def print_config(self):
        print("\n=== Baseline ViT Prototype Config ===")
        print(f"Project: {self.args.project}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Data root: {self.args.dataroot}")
        print(f"Model: {self.args.model}")
        print(f"Base classes: {self.args.base_class}, total classes: {self.args.num_classes}")
        print(f"Way/shot/sessions: {self.args.way}/{self.args.shot}/{self.args.sessions}")
        pretrain_lr = self.args.lr_pretrain if self.args.lr_pretrain is not None else self.args.lr_base
        print(
            f"Epochs: base_pretrain={self.args.base_pretrain_epochs}, "
            f"base={self.args.epochs_base}, incremental={self.args.epochs_joint}"
        )
        print(
            f"Incremental CE weights: base/old={self.args.incremental_base_loss_weight}, "
            f"all_novel={self.args.incremental_new_loss_weight}"
        )
        print(f"Save all session checkpoints: {self.args.save_all_sessions}")
        print(f"Batch sizes: base={self.args.batch_size_base}, train_base={self.args.batch_size_train_base}, replay={self.args.batch_size_replay}, test={self.args.batch_size_test}")
        print(f"Learning rates: pretrain={pretrain_lr}, base={self.args.lr_base}, incremental={self.args.lr_new}")
        print(
            f"LR scheduler: {self.args.lr_scheduler}, min_lr_base={self.args.min_lr_base}, "
            f"min_lr_new={self.args.min_lr_new}, warmup_start_lr={self.args.warmup_start_lr}"
        )
        print(
            f"Warmup epochs: pretrain={self.args.warmup_epochs_pretrain}, "
            f"base={self.args.warmup_epochs_base}, incremental={self.args.warmup_epochs_new}"
        )
        print(
            f"Feature extractor: {self.args.feature_extractor}, forward_layers={self.args.forward_layers}, "
            f"tokens={self.args.forward_token_nums}, hidden={self.args.forward_hidden_dim}, "
            f"active_attn_hidden={self.args.forward_active_attn_hidden_dim}, "
            f"active_ablation={self.args.active_ablation}"
        )
        print(
            f"Incremental prompt context: layer-wise H, "
            f"mode={self.args.prompt_mode}, "
            f"hidden={self.args.incremental_prompt_context_hidden_dim}, "
            f"scale={self.args.incremental_prompt_context_scale}"
        )
        print(
            f"Combine: hidden={self.args.combine_hidden_dim}, drop={self.args.combine_drop}, "
            f"ablation={self.args.combine_ablation}, fusion={self.args.fusion_method}"
        )
        print(f"Classifier: {self.args.classifier}, prototype context weight={self.args.proto_context_weight}")
        if self.args.classifier == "orco_fagg_mab_project":
            print(
                f"OrCo: temperature={self.args.orco_temperature}, reserve={self.args.orco_reserve_mode}, "
                f"supcon_temperature={self.args.orco_supcon_temperature}, "
                f"sup={self.args.orco_sup_lam}, cos={self.args.orco_cos_lam}, simplex={self.args.orco_simplex_lam}, "
                f"target_epochs={self.args.orco_target_epochs}"
            )
        print(
            f"MAB head: heads={self.args.mab_num_heads}, hidden={self.args.mab_hidden_dim}, "
            f"drop={self.args.mab_drop}, attn_drop={self.args.mab_attn_drop}, "
            f"res_scale={self.args.mab_res_scale}"
        )
        print(f"Prototype temperature: {self.args.proto_temperature}")
        print(f"Local ViT path: {self.args.vit_pretrained_path}")
        print("====================================\n")

    def build_model(self):
        model = BaselineViTProtoNet(self.args, mode=self.args.base_mode)
        model = nn.DataParallel(model, list(range(self.args.num_gpu)))
        return model.cuda()

    def set_save_path(self):
        self.args.save_path = osp.join("checkpoint_wd0.0", self.args.dataset, self.args.project)
        if self.args.save_path_prefix:
            self.args.save_path = osp.join(self.args.save_path, self.args.save_path_prefix)
        utils.ensure_path(self.args.save_path)

    def make_output_dir(self):
        run_name = datetime.datetime.now().__format__("%m-%d-%H-%M-%S") + "_baseline_proto"
        self.args.output_dir = osp.join(self.args.save_path, run_name)
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        save_args_to_yaml(self.args, osp.join(self.args.output_dir, "config.yaml"))
        print(f"Output dir: {self.args.output_dir}")

    def get_dataloader(self, session):
        if session == 0:
            return data_utils.get_base_dataloader(self.args)
        return data_utils.get_new_dataloader(self.args, session)

    def get_optimizer(self, lr, weight_decay):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.args.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if self.args.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    def get_lr_scheduler(self, optimizer, epochs, min_lr, warmup_epochs):
        if self.args.lr_scheduler == "none":
            return None
        if self.args.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs),
                eta_min=min_lr,
            )
        if self.args.lr_scheduler == "warmup_cosine":
            max_epochs = max(3, epochs)
            safe_warmup_epochs = min(max(2, warmup_epochs), max_epochs - 1)
            return LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=safe_warmup_epochs,
                max_epochs=max_epochs,
                warmup_start_lr=self.args.warmup_start_lr,
                eta_min=min_lr,
            )
        raise ValueError(f"Unsupported lr_scheduler: {self.args.lr_scheduler}")

    def get_current_lr(self, optimizer):
        return optimizer.param_groups[0]["lr"]

    def ensure_model_input_size(self, images):
        target_size = getattr(self.model.module.encoder.patch_embed, "img_size", None)
        if target_size is None:
            return images
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        if images.shape[-2:] == tuple(target_size):
            return images
        return F.interpolate(images, size=tuple(target_size), mode="bilinear", align_corners=False)

    def unpack_train_batch(self, batch):
        images, labels = batch
        base_labels = labels.cuda()
        if isinstance(images, (list, tuple)):
            views = [view.cuda() for view in images]
            images = torch.cat(views, dim=0)
            labels = base_labels.repeat(len(views))
            nviews = len(views)
        else:
            images = images.cuda()
            labels = base_labels
            nviews = 1
        images = self.ensure_model_input_size(images)
        return images, labels, base_labels, nviews

    def compute_cls_loss(self, logits, labels, session):
        if (
            session == 0
            or (
                self.args.incremental_new_loss_weight == 1.0
                and self.args.incremental_base_loss_weight == 1.0
            )
        ):
            return F.cross_entropy(logits, labels)

        base_mask = labels < self.args.base_class
        novel_mask = labels >= self.args.base_class
        loss = logits.new_tensor(0.0)

        if base_mask.any() and self.args.incremental_base_loss_weight != 0.0:
            loss = loss + self.args.incremental_base_loss_weight * F.cross_entropy(
                logits[base_mask],
                labels[base_mask],
            )
        if novel_mask.any() and self.args.incremental_new_loss_weight != 0.0:
            loss = loss + self.args.incremental_new_loss_weight * F.cross_entropy(
                logits[novel_mask],
                labels[novel_mask],
            )

        return loss

    def get_orco_pscl_targets(self, session, device):
        fc = self.model.module.fc
        unassigned_targets = fc.get_unassigned_targets().detach().clone()

        if session == 0:
            target_prototypes = unassigned_targets
            target_labels = torch.arange(
                self.args.base_class,
                self.args.base_class + target_prototypes.shape[0],
                device=device,
            )
            return target_prototypes, target_labels

        assigned_targets = fc.get_classifier_weights().detach().clone()
        assigned_labels = fc.get_classifier_labels().detach().clone().to(device)
        novel_mask = assigned_labels >= self.args.base_class

        pieces = []
        labels = []
        if novel_mask.any():
            pieces.append(assigned_targets[novel_mask])
            labels.append(assigned_labels[novel_mask])
        if unassigned_targets.numel() > 0:
            pieces.append(unassigned_targets)
            future_start = self.args.base_class + int(novel_mask.sum().item())
            labels.append(
                torch.arange(
                    future_start,
                    future_start + unassigned_targets.shape[0],
                    device=device,
                )
            )

        if not pieces:
            return torch.empty(0, self.args.proj_output_dim, device=device), torch.empty(0, device=device, dtype=torch.long)

        return torch.cat(pieces, dim=0), torch.cat(labels, dim=0)

    def compute_orco_loss(self, logits, features, labels, base_labels, nviews, seen_classes, session):
        features = normalize(features)
        cls_loss = self.compute_cls_loss(logits, labels, session)
        loss = self.args.orco_cos_lam * cls_loss

        if features.shape[0] % nviews == 0:
            split_size = features.shape[0] // nviews
            split_features = torch.split(features, split_size, dim=0)
            target_prototypes, target_labels = self.get_orco_pscl_targets(session, features.device)
            perturbed_targets, target_labels = perturb_targets_norm_count(
                target_prototypes,
                target_labels,
                base_labels.shape[0],
                nviews=nviews,
                epsilon=self.args.orco_perturb_epsilon,
                offset=self.args.orco_perturb_offset,
            )
            features_add_targets = []
            for view_idx in range(nviews):
                features_add_targets.append(
                    torch.cat((split_features[view_idx], perturbed_targets[view_idx]), dim=0).unsqueeze(1)
                )
            features_add_targets = torch.cat(features_add_targets, dim=1)
            supcon_labels = torch.cat((base_labels, target_labels), dim=0)
            sup_loss = SupConLoss(
                temperature=self.args.orco_supcon_temperature,
                base_temperature=self.args.orco_supcon_temperature,
            )(
                features_add_targets,
                supcon_labels,
            )
            loss = loss + self.args.orco_sup_lam * sup_loss
        else:
            sup_loss = features.new_tensor(0.0)

        assigned_targets = self.model.module.fc.get_classifier_weights().detach().clone()
        assigned_labels = self.model.module.fc.get_classifier_labels().detach().clone()
        unassigned_targets = self.model.module.fc.get_unassigned_targets().detach().clone()
        if session > 0:
            new_ixs = labels >= self.args.base_class
            simplex_features = features[new_ixs]
            simplex_labels = labels[new_ixs]
        else:
            simplex_features = features
            simplex_labels = labels
        orth_loss = simplex_loss(simplex_features, simplex_labels, assigned_targets, assigned_labels, unassigned_targets)
        loss = loss + self.args.orco_simplex_lam * orth_loss
        return loss, cls_loss, sup_loss, orth_loss

    def train_epoch(self, loader, optimizer, seen_classes, desc, session=0):
        self.model.train()
        loss_meter = utils.Averager()
        cls_loss_meter = utils.Averager()
        sup_loss_meter = utils.Averager()
        orth_loss_meter = utils.Averager()
        acc_meter = utils.Averager()

        tqdm_gen = tqdm(loader, desc=desc, dynamic_ncols=True)
        for batch_idx, batch in enumerate(tqdm_gen, 1):
            images, labels, base_labels, nviews = self.unpack_train_batch(batch)
            logits, features = self.model(images)
            logits = logits[:, :seen_classes]
            if self.args.classifier == "orco_fagg_mab_project":
                loss, cls_loss, sup_loss, orth_loss = self.compute_orco_loss(
                    logits,
                    features,
                    labels,
                    base_labels,
                    nviews,
                    seen_classes,
                    session,
                )
            else:
                cls_loss = self.compute_cls_loss(logits, labels, session)
                sup_loss = logits.new_tensor(0.0)
                orth_loss = logits.new_tensor(0.0)
                loss = cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            cls_loss_meter.add(cls_loss.item())
            sup_loss_meter.add(sup_loss.item())
            orth_loss_meter.add(orth_loss.item())
            acc_meter.add(utils.count_acc(logits, labels))
            postfix = {
                "batch": f"{batch_idx}/{len(loader)}",
                "seen": seen_classes,
                "loss": f"{loss_meter.item():.4f}",
                "cls": f"{cls_loss_meter.item():.4f}",
                "acc": f"{acc_meter.item() * 100:.2f}",
            }
            if self.args.classifier == "orco_fagg_mab_project":
                postfix["sup"] = f"{sup_loss_meter.item():.4f}"
                postfix["orth"] = f"{orth_loss_meter.item():.4f}"
            tqdm_gen.set_postfix(**postfix, refresh=False)

        return loss_meter.item(), acc_meter.item()

    def train_base_pretrain_epoch(self, loader, optimizer, desc):
        self.model.train()
        loss_meter = utils.Averager()
        tqdm_gen = tqdm(loader, desc=desc, dynamic_ncols=True)

        for batch_idx, batch in enumerate(tqdm_gen, 1):
            images, _, base_labels, nviews = self.unpack_train_batch(batch)
            _, features = self.model(images)
            features = normalize(features)

            if features.shape[0] % nviews != 0:
                raise RuntimeError(
                    f"SupCon pretrain expected feature batch divisible by nviews, "
                    f"got features={features.shape[0]}, nviews={nviews}."
                )

            split_size = features.shape[0] // nviews
            split_features = torch.split(features, split_size, dim=0)
            supcon_features = torch.stack(split_features, dim=1)
            loss = SupConLoss(
                temperature=self.args.orco_supcon_temperature,
                base_temperature=self.args.orco_supcon_temperature,
            )(supcon_features, base_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            tqdm_gen.set_postfix(
                batch=f"{batch_idx}/{len(loader)}",
                sup=f"{loss_meter.item():.4f}",
                refresh=False,
            )

        return loss_meter.item()

    @torch.no_grad()
    def update_seen_prototypes(self, loader, seen_classes):
        class_list = np.arange(seen_classes)
        print(f"Updating prototypes for seen classes [0, {seen_classes - 1}] from {len(loader.dataset)} samples...")
        stats = self.model.module.update_prototypes(loader, class_list)
        print(f"Updated {len(stats['updated_classes'])}/{seen_classes} prototypes.")
        return stats

    def build_deterministic_loader_like_test(self, dataset, testloader, shuffle=False):
        stable_dataset = deepcopy(dataset)
        stable_dataset.transform = testloader.dataset.transform
        stable_loader = torch.utils.data.DataLoader(
            dataset=stable_dataset,
            batch_size=self.args.batch_size_test,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return stable_loader

    def save_checkpoint(self, session):
        save_model_dir = osp.join(self.args.output_dir, f"session{session}_max_acc.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all(),
                "numpy_random_state": np.random.get_state(),
                "python_random_state": random.getstate(),
            },
            save_model_dir,
        )
        print(f"===[Session-{session}] Saving model to: {save_model_dir}===")

    def should_save_checkpoint(self, session):
        return self.args.save_all_sessions or session == 0 or session == self.args.sessions - 1

    def update_best_model(self, session):
        self.best_model_dict = deepcopy(self.model.state_dict())
        if self.should_save_checkpoint(session):
            self.save_checkpoint(session)
        else:
            print(f"===[Session-{session}] Skipping checkpoint save; keeping model state in memory===")

    def record_session_result(self, session, acc, novel_acc, base_acc):
        hm = utils.hm(base_acc, novel_acc) if session > 0 else 0.0
        result = {
            "session": session,
            "way": self.args.way,
            "shot": self.args.shot,
            "seen_classes": self.args.base_class + session * self.args.way,
            "top1_acc": acc * 100,
            "base_acc": base_acc * 100,
            "novel_acc": novel_acc * 100,
            "hm": hm * 100,
        }

        self.session_results = [item for item in self.session_results if item["session"] != session]
        self.session_results.append(result)
        self.session_results.sort(key=lambda item: item["session"])
        running_acc = 0.0
        for idx, item in enumerate(self.session_results, 1):
            running_acc += item["top1_acc"]
            item["avg_acc"] = running_acc / idx
        self.save_session_results()
        return next(item for item in self.session_results if item["session"] == session)

    def save_session_results(self):
        output_file = osp.join(self.args.output_dir, "session_results.csv")
        header = "session,way,shot,seen_classes,top1_acc,base_acc,novel_acc,hm,avg_acc\n"
        lines = [header]
        for item in self.session_results:
            lines.append(
                "{session},{way},{shot},{seen_classes},{top1_acc:.3f},{base_acc:.3f},{novel_acc:.3f},{hm:.3f},{avg_acc:.3f}\n".format(
                    **item
                )
            )
        with open(output_file, "w") as f:
            f.writelines(lines)
        print(f"Session accuracy saved to: {output_file}")

    def record_epoch_log(
        self,
        session,
        epoch,
        phase,
        lr,
        train_loss,
        train_acc,
        test_acc,
        novel_acc,
        base_acc,
        hm,
    ):
        self.epoch_logs.append(
            {
                "session": session,
                "epoch": epoch,
                "phase": phase,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc * 100,
                "test_acc": test_acc * 100,
                "base_acc": base_acc * 100,
                "novel_acc": novel_acc * 100,
                "hm": hm * 100,
            }
        )
        self.save_epoch_logs()

    def save_epoch_logs(self):
        output_file = osp.join(self.args.output_dir, "logs.csv")
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        header = "session,epoch,phase,lr,train_loss,train_acc,test_acc,base_acc,novel_acc,hm\n"
        lines = [header]
        for item in self.epoch_logs:
            lines.append(
                "{session},{epoch},{phase},{lr:.8f},{train_loss:.6f},{train_acc:.3f},{test_acc:.3f},{base_acc:.3f},{novel_acc:.3f},{hm:.3f}\n".format(
                    **item
                )
            )
        with open(output_file, "w") as f:
            f.writelines(lines)

    def train_base(self):
        print("\n===[Base Session] Preparing dataloaders===")
        self.model.module.set_base_prompts_trainable(True)
        base_set, _, base_testloader = self.get_dataloader(0)
        _, base_trainloader, _ = data_utils.get_baseline_base_train_dataloader(self.args)
        proto_loader = torch.utils.data.DataLoader(
            dataset=base_set,
            batch_size=self.args.batch_size_test,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        proto_loader.dataset.transform = base_testloader.dataset.transform

        seen_classes = self.args.base_class
        print(f"Base train samples: {len(base_trainloader.dataset)}, prototype samples: {len(proto_loader.dataset)}, test samples: {len(base_testloader.dataset)}")
        if self.args.classifier == "orco_fagg_mab_project":
            print(
                f"Generating OrCo reserve vectors: epochs={self.args.orco_target_epochs}, "
                f"lr={self.args.orco_target_lr}"
            )
            self.model.module.fc.find_reserve_vectors_all(
                epochs=self.args.orco_target_epochs,
                lr=self.args.orco_target_lr,
            )
        self.update_seen_prototypes(proto_loader, seen_classes)

        optimizer = self.get_optimizer(self.args.lr_base, self.args.decay)
        scheduler = self.get_lr_scheduler(
            optimizer,
            self.args.epochs_base,
            self.args.min_lr_base,
            self.args.warmup_epochs_base,
        )
        self.model.module.get_trainable_params()
        best_acc = -1.0
        best_state_dict = None
        best_eval = None

        for epoch in range(self.args.epochs_base):
            current_lr = self.get_current_lr(optimizer)
            print(f"\n[Base] Epoch {epoch + 1}/{self.args.epochs_base} | lr={current_lr:.6f}")
            train_loss, train_acc = self.train_epoch(
                base_trainloader,
                optimizer,
                seen_classes,
                f"[Base][Epoch {epoch + 1}/{self.args.epochs_base}]",
                session=0,
            )
            self.update_seen_prototypes(proto_loader, seen_classes)
            acc, novel_cw, base_cw = evaluate(self.model, base_testloader, self.args, 0)
            self.record_epoch_log(
                session=0,
                epoch=epoch + 1,
                phase="base",
                lr=current_lr,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=acc,
                novel_acc=novel_cw,
                base_acc=base_cw,
                hm=0.0,
            )
            print(
                f"[Base] Epoch {epoch + 1}/{self.args.epochs_base} summary | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}, "
                f"test_acc={acc * 100:.2f}, base_acc={base_cw * 100:.2f}"
            )
            if acc > best_acc:
                best_acc = acc
                best_eval = (acc, novel_cw, base_cw)
                best_state_dict = deepcopy(self.model.state_dict())
                print(f"[Base] New best epoch {epoch + 1}: test_acc={acc * 100:.2f}")
            if scheduler is not None:
                scheduler.step()

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict, strict=True)
            acc, novel_cw, base_cw = best_eval
        else:
            acc, novel_cw, base_cw = evaluate(self.model, base_testloader, self.args, 0)
        self.trlog["max_acc"][0] = float("%.3f" % (acc * 100))
        self.trlog["max_base_acc"][0] = float("%.3f" % (base_cw * 100))
        self.trlog["max_novel_acc"][0] = float("%.3f" % (novel_cw * 100))
        result = self.record_session_result(0, acc, novel_cw, base_cw)
        print(f"[Base] Acc: {acc * 100:.3f}, avg_acc: {result['avg_acc']:.3f}")
        self.update_best_model(0)

    def train_base_pretrain(self):
        if self.args.base_pretrain_epochs <= 0:
            return

        print("\n===[Phase-1][Base SupCon Pretrain] Preparing dataloaders===")
        self.model.module.set_base_prompts_trainable(True)
        _, base_trainloader, _ = data_utils.get_baseline_base_train_dataloader(self.args)
        lr_pretrain = self.args.lr_pretrain if self.args.lr_pretrain is not None else self.args.lr_base
        optimizer = self.get_optimizer(lr_pretrain, self.args.decay)
        scheduler = self.get_lr_scheduler(
            optimizer,
            self.args.base_pretrain_epochs,
            self.args.min_lr_pretrain,
            self.args.warmup_epochs_pretrain,
        )
        self.model.module.get_trainable_params()
        print(
            f"Base SupCon pretrain samples: {len(base_trainloader.dataset)}, "
            f"batches: {len(base_trainloader)}, epochs: {self.args.base_pretrain_epochs}"
        )

        for epoch in range(self.args.base_pretrain_epochs):
            current_lr = self.get_current_lr(optimizer)
            print(
                f"\n[Base Pretrain] Epoch {epoch + 1}/{self.args.base_pretrain_epochs} "
                f"| lr={current_lr:.6f}"
            )
            train_loss = self.train_base_pretrain_epoch(
                base_trainloader,
                optimizer,
                f"[Base Pretrain][Epoch {epoch + 1}/{self.args.base_pretrain_epochs}]",
            )
            self.record_epoch_log(
                session=0,
                epoch=epoch + 1,
                phase="base_pretrain",
                lr=current_lr,
                train_loss=train_loss,
                train_acc=0.0,
                test_acc=0.0,
                novel_acc=0.0,
                base_acc=0.0,
                hm=0.0,
            )
            print(f"[Base Pretrain] Epoch {epoch + 1}/{self.args.base_pretrain_epochs} summary | sup_loss={train_loss:.4f}")
            if scheduler is not None:
                scheduler.step()

    def train_incremental_session(self, session):
        self.model.load_state_dict(self.best_model_dict, strict=True)

        print(f"\n===[Incremental Session {session}] Preparing dataloaders===")
        train_set, trainloader, testloader = self.get_dataloader(session)
        seen_classes = self.args.base_class + session * self.args.way

        print(f"\n\n===[Incremental][Session-{session}] Started!===")
        new_classes = np.unique(train_set.targets)
        print(f"New classes: {new_classes.tolist()}")
        print(f"New train samples: {len(trainloader.dataset)}, test samples: {len(testloader.dataset)}, seen classes: {seen_classes}")
        new_start = self.args.base_class + (session - 1) * self.args.way
        new_end = self.args.base_class + session * self.args.way
        print(
            f"Current-session new-class range: [{new_start}, {new_end - 1}], "
            f"ce_base_weight={self.args.incremental_base_loss_weight}, "
            f"ce_all_novel_weight={self.args.incremental_new_loss_weight}"
        )
        stable_new_loader = self.build_deterministic_loader_like_test(train_set, testloader)
        context = self.model.module.update_incremental_prompt_context(stable_new_loader, new_classes)
        if context is None:
            print("Layer-wise H prompt context was not set.")
        else:
            print(
                f"Layer-wise H prompt context set: shape={tuple(context.shape)}, "
                f"norm={context.norm().item():.4f}"
            )
        self.model.module.set_base_prompts_trainable(False)
        print("Frozen base prompts for incremental training.")
        stats = self.model.module.update_prototypes(stable_new_loader, new_classes)
        print(f"Initialized {len(stats['updated_classes'])} new-class prototypes.")

        replay_set, replay_loader = data_utils.get_baseline_replay_dataloader(self.args, session)
        print(f"Replay samples: {len(replay_loader.dataset)}, replay batches: {len(replay_loader)}")
        print("Pre-updating all seen prototypes after prompt context setup...")
        self.update_seen_prototypes(replay_loader, seen_classes)
        optimizer = self.get_optimizer(self.args.lr_new, self.args.decay_new)
        scheduler = self.get_lr_scheduler(
            optimizer,
            self.args.epochs_joint,
            self.args.min_lr_new,
            self.args.warmup_epochs_new,
        )
        self.model.module.get_trainable_params()

        for epoch in range(self.args.epochs_joint):
            current_lr = self.get_current_lr(optimizer)
            print(f"\n[Session {session}] Epoch {epoch + 1}/{self.args.epochs_joint} | lr={current_lr:.6f}")
            train_loss, train_acc = self.train_epoch(
                replay_loader,
                optimizer,
                seen_classes,
                f"[Session {session}][Epoch {epoch + 1}/{self.args.epochs_joint}]",
                session=session,
            )
            self.update_seen_prototypes(replay_loader, seen_classes)
            tsa, novel_cw, base_cw = evaluate(self.model, testloader, self.args, session)
            hm = utils.hm(base_cw, novel_cw)
            self.record_epoch_log(
                session=session,
                epoch=epoch + 1,
                phase="incremental",
                lr=current_lr,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=tsa,
                novel_acc=novel_cw,
                base_acc=base_cw,
                hm=hm,
            )
            print(
                f"[Session {session}] Epoch {epoch + 1}/{self.args.epochs_joint} summary | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}, "
                f"test_acc={tsa * 100:.2f}, novel_acc={novel_cw * 100:.2f}, "
                f"base_acc={base_cw * 100:.2f}, hm={hm * 100:.2f}"
            )
            if scheduler is not None:
                scheduler.step()

        tsa, novel_cw, base_cw = evaluate(self.model, testloader, self.args, session)
        self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
        self.trlog["max_novel_acc"][session] = float("%.3f" % (novel_cw * 100))
        self.trlog["max_base_acc"][session] = float("%.3f" % (base_cw * 100))
        self.trlog["max_hm"][session] = float("%.3f" % (utils.hm(base_cw, novel_cw) * 100))
        result = self.record_session_result(session, tsa, novel_cw, base_cw)

        self.update_best_model(session)
        print(
            "Session {}, test Acc {:.3f}, test_novel_acc {:.3f}, test_base_acc {:.3f}, hm {:.3f}, avg_acc {:.3f}".format(
                session,
                self.trlog["max_acc"][session],
                self.trlog["max_novel_acc"][session],
                self.trlog["max_base_acc"][session],
                self.trlog["max_hm"][session],
                result["avg_acc"],
            )
        )

    def exit_log(self, result_list):
        if self.trlog["max_hm"]:
            self.trlog["max_hm"][0] = 0.0

        result_list.append("Top 1 Accuracy: ")
        result_list.append(self.trlog["max_acc"])
        result_list.append("Harmonic Mean: ")
        result_list.append(self.trlog["max_hm"][1:])
        result_list.append("Base Test Accuracy: ")
        result_list.append(self.trlog["max_base_acc"])
        result_list.append("Novel Test Accuracy: ")
        result_list.append(self.trlog["max_novel_acc"])

        valid_hm = np.array(self.trlog["max_hm"][1:]) if self.args.sessions > 1 else np.array([0.0])
        average_harmonic_mean = valid_hm.mean()
        average_acc = np.array(self.trlog["max_acc"]).mean()
        performance_decay = self.trlog["max_acc"][0] - self.trlog["max_acc"][-1]

        result_list.append("Average Harmonic Mean Accuracy: ")
        result_list.append(average_harmonic_mean)
        result_list.append("Average Accuracy: ")
        result_list.append(average_acc)
        result_list.append("Performance Decay: ")
        result_list.append(performance_decay)

        print(f"\n\nacc: {self.trlog['max_acc']}")
        print(f"avg_acc: {average_acc:.3f}")
        print(f"hm: {self.trlog['max_hm'][1:]}")
        print(f"avg_hm: {average_harmonic_mean:.3f}")
        print(f"pd: {performance_decay:.3f}")
        print(f"base: {self.trlog['max_base_acc']}")
        print(f"novel: {self.trlog['max_novel_acc']}")
        utils.save_list_to_txt(osp.join(self.args.output_dir, "results.txt"), result_list)

    def train(self):
        t_start_time = time.time()
        self.make_output_dir()
        result_list = [
            self.args,
            f"Dataset {self.args.dataset}, way {self.args.way}, shot {self.args.shot}, "
            f"base_class {self.args.base_class}, num_classes {self.args.num_classes}, sessions {self.args.sessions}, "
            f"save_all_sessions {self.args.save_all_sessions}, lr_scheduler {self.args.lr_scheduler}, "
            f"base_pretrain_epochs {self.args.base_pretrain_epochs}, "
            f"warmup_epochs_pretrain {self.args.warmup_epochs_pretrain}, "
            f"warmup_epochs_base {self.args.warmup_epochs_base}, warmup_epochs_new {self.args.warmup_epochs_new}",
        ]

        self.train_base_pretrain()
        self.train_base()
        result_list.append(
            f"Session 0, test Acc {self.trlog['max_acc'][0]:.3f}, avg_acc {self.trlog['max_acc'][0]:.3f}"
        )

        for session in range(1, self.args.sessions):
            self.train_incremental_session(session)
            avg_acc = np.array(self.trlog["max_acc"][: session + 1]).mean()
            result_list.append(
                "Session {}, test Acc {:.3f}, test_novel_acc {:.3f}, test_base_acc {:.3f}, hm {:.3f}, avg_acc {:.3f}".format(
                    session,
                    self.trlog["max_acc"][session],
                    self.trlog["max_novel_acc"][session],
                    self.trlog["max_base_acc"][session],
                    self.trlog["max_hm"][session],
                    avg_acc,
                )
            )

        self.exit_log(result_list)
        total_time = (time.time() - t_start_time) / 60
        print("Total time used %.3f mins" % total_time)

    def test(self):
        output_file_path = osp.join(self.args.output_dir, "session_acc.txt")
        result_list = []
        self.session_results = []

        for session in range(self.args.sessions):
            checkpoint_path = osp.join(self.args.output_dir, f"session{session}_max_acc.pth")
            if not osp.exists(checkpoint_path):
                print(f"Skip session {session}: checkpoint not found at {checkpoint_path}")
                continue

            self.model = self.build_model()
            checkpoint = torch.load(checkpoint_path)
            load_info = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if load_info.missing_keys or load_info.unexpected_keys:
                print(
                    f"Checkpoint loaded with non-strict keys | "
                    f"missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}"
                )

            _, _, testloader = self.get_dataloader(session)
            tsa, novel_cw, base_cw = evaluate(self.model, testloader, self.args, session)
            hm = utils.hm(base_cw, novel_cw)

            result = self.record_session_result(session, tsa, novel_cw, base_cw)

            out_string = (
                f"Session {session} - Top-1 Acc: {tsa * 100:.3f}, "
                f"Novel Acc: {novel_cw * 100:.3f}, Base Acc: {base_cw * 100:.3f}, "
                f"HM: {hm * 100:.3f}, Avg Acc: {result['avg_acc']:.3f}"
            )
            print(out_string)
            result_list.append(out_string)

        utils.save_list_to_txt(output_file_path, result_list)
