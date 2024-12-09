from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def kd_normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    assert has_distributed, "torch.distributed did not import correctly, please use a PyTorch version with support."
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def gather_features_single(
    image_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(
            torch.distributed.nn.all_gather(image_features), dim=0
        )

    else:
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]

        dist.all_gather(gathered_image_features, image_features)

        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features

        all_image_features = torch.cat(gathered_image_features, dim=0)

    return all_image_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        output_dict=False,
        return_logits=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        if return_logits:
            return {
                "contrastive_loss": total_loss,
                "logits_per_image": logits_per_image,
            } if output_dict else total_loss, logits_per_image

        return {"contrastive_loss": total_loss} if output_dict else total_loss


# def KD_Norm_Loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
#     logits_student = (
#         kd_normalize(logits_student_in) if logit_stand else logits_student_in
#     )
#     logits_teacher = (
#         kd_normalize(logits_teacher_in) if logit_stand else logits_teacher_in
#     )
#     log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
#     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
#     loss_kd *= temperature**2
#     return loss_kd


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left]
    )
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(
            left_rank, right_rank, tensor_to_left, tensor_to_right, group=group
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
        )


def neighbour_exchange_bidir_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    return NeighbourExchangeBidir.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        cache_labels=False,
        rank=0,
        world_size=1,
        bidir=True,
        use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(
        self, device, dtype, num_logits, negative_only=False
    ) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        negative_only=False,
    ):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        output_dict=False,
    ):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class KD_Norm_Loss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad

        self.rank = rank
        self.world_size = world_size
        self.T = nn.Parameter(torch.tensor(1.0))

    def distill_loss(
        self, logits_student_in, logits_teacher_in, temperature, logit_stand=True
    ):
        logits_student = (
            kd_normalize(logits_student_in) if logit_stand else logits_student_in
        )
        logits_teacher = (
            kd_normalize(logits_teacher_in) if logit_stand else logits_teacher_in
        )
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = (
            F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        )
        loss_kd *= temperature**2
        return loss_kd

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                False,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(
        self,
        student_image_features,
        student_text_features,
        student_logit_scale,
        teacher_image_features,
        teacher_text_features,
        teacher_logit_scale,
        output_dict=False,
    ):
        student_logits_per_image, student_logits_per_text = self.get_logits(
            student_image_features, student_text_features, student_logit_scale
        )

        teacher_logits_per_image, teacher_logits_per_text = self.get_logits(
            teacher_image_features, teacher_text_features, teacher_logit_scale
        )

        total_loss = (
            self.distill_loss(
                student_logits_per_image, teacher_logits_per_image, self.T
            )
            + self.distill_loss(
                student_logits_per_text, teacher_logits_per_text, self.T
            )
        ) / 2

        return {"distill_loss": total_loss} if output_dict else total_loss


def single_uni_modal_clip_loss_gather(
    cur_modal,
    samples_proj2text_features,
    samples_text_features,
    logit_scale,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    args=None,
):
    dict_nce_loss = {}
    sim_dict = {}
    
    if args.use_sigliploss:
        clip_loss = SigLipLoss(cache_labels=True,rank=rank,world_size=world_size)
    else:
        clip_loss = ClipLoss(local_loss=local_loss,gather_with_grad=gather_with_grad,cache_labels=True,rank=rank,world_size=world_size)
    
    return_logits=False
    if len(samples_proj2text_features[cur_modal]) > 1:
        return_logits = True
        kd_norm_loss = DistillKL(T=1.0, logit_stand=True)

    for anchor, _ in samples_proj2text_features[cur_modal].items():
        if return_logits:
            nce_anchor2text_loss, sim_anchor2text = clip_loss(samples_proj2text_features[cur_modal][anchor],samples_text_features[cur_modal],logit_scale[anchor],return_logits=return_logits)
            sim_dict[anchor] = sim_anchor2text
        else:
            nce_anchor2text_loss = clip_loss(samples_proj2text_features[cur_modal][anchor],samples_text_features[cur_modal],logit_scale[anchor],return_logits=return_logits)

        dict_nce_loss.update({f"nce_loss_{cur_modal}_{anchor}2t": nce_anchor2text_loss})

        if len(samples_proj2text_features[cur_modal]) > 1:
        
            for target, _ in samples_proj2text_features[cur_modal].items():
                if anchor != target and anchor != cur_modal:
                    
                    nce_anchor2target_loss= clip_loss(samples_proj2text_features[cur_modal][anchor],samples_proj2text_features[cur_modal][target],logit_scale[anchor])

                    dict_nce_loss.update(
                        {f"nce_loss_{cur_modal}_{anchor}2{target}": nce_anchor2target_loss}
                    )
        
    if len(sim_dict) > 1:
        # kd_loss=0
        for k, v in sim_dict.items():
            if k != 'image':
                kd_loss = kd_norm_loss(v, sim_dict['image'])
            dict_nce_loss.update({f"uni_kd_loss_{cur_modal}2_{k}": kd_loss})        

    return dict_nce_loss


def single_cross_modal_kd_loss_gather(
    cur_modal,
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    modal_labels_features,
    logit_scale,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    dict_kd_loss = {}

    # text_features = gather_features_single(
    #     samples_text_features[cur_modal], local_loss, gather_with_grad, rank, world_size
    # )
    kd_norm_loss = KD_Norm_Loss(local_loss, gather_with_grad, rank, world_size)

    for m_target in modal_list:
        for anchor, _ in samples_proj2text_features[cur_modal].items():
            if m_target != cur_modal:

                kd_loss = kd_norm_loss(samples_proj2text_features[cur_modal][anchor],modal_labels_features[m_target],logit_scale[cur_modal],samples_text_features[cur_modal],modal_labels_features[m_target], logit_scale[cur_modal])

                if anchor == cur_modal:
                    dict_kd_loss.update(
                        {f"cross_kd_loss_{cur_modal}2{m_target}": kd_loss}
                    )
                else:
                    dict_kd_loss.update(
                        {f"cross_kd_loss_{cur_modal}_{anchor}2{m_target}": kd_loss}
                    )
    # for m_target in modal_list:
    #     if m_target != cur_modal:
    #         kd_loss = kd_norm_loss(
    #             samples_proj2text_features[cur_modal][cur_modal],
    #             modal_labels_features[m_target],
    #             logit_scale[cur_modal],
    #             samples_text_features[cur_modal],
    #             modal_labels_features[m_target],
    #             logit_scale[cur_modal],
    #         )

    #         dict_kd_loss.update({f"cross_kd_loss_{cur_modal}2{m_target}": kd_loss})

    return dict_kd_loss


def single_cross_modal_kd_loss_visual(
    cur_modal,
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    logit_scale,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    dict_kd_loss = {}

    # text_features = gather_features_single(
    #     samples_text_features[cur_modal], local_loss, gather_with_grad, rank, world_size
    # )
    kd_norm_loss = KD_Norm_Loss(local_loss, gather_with_grad, rank, world_size)

    for m_target in modal_list:
        if m_target != cur_modal:
            kd_loss = kd_norm_loss(
                samples_proj2text_features[cur_modal][cur_modal],
                samples_proj2text_features[m_target][m_target],
                logit_scale[cur_modal],
                samples_text_features[cur_modal],
                samples_text_features[m_target],
                logit_scale[cur_modal],
            )

            dict_kd_loss.update({f"cross_kd_loss_{cur_modal}2{m_target}": kd_loss})

    return dict_kd_loss


def single_cross_modal_kd_loss_plm(
    cur_modal,
    modal_list,
    samples_proj2text_features,
    modal_labels_features,
    plm_text_features,
    plm_labels_features,
    logit_scale,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    dict_kd_loss = {}

    # text_features = gather_features_single(
    #     samples_text_features[cur_modal], local_loss, gather_with_grad, rank, world_size
    # )
    kd_norm_loss = KD_Norm_Loss(local_loss, gather_with_grad, rank, world_size)

    for m_target in modal_list:
        if m_target != cur_modal:
            kd_loss = kd_norm_loss(
                samples_proj2text_features[cur_modal][cur_modal],
                modal_labels_features[m_target],
                logit_scale[cur_modal],
                plm_text_features,
                plm_labels_features[m_target],
                logit_scale[cur_modal],
            )

            dict_kd_loss.update({f"cross_kd_loss_{cur_modal}2{m_target}": kd_loss})

    return dict_kd_loss


def cross_modal_cls_gather(
    cur_modal,
    samples_proj2text_features,
    modal_labels_features,
    targets,
    logit_scale,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    # if cur_modal == "audio":
    #     targets = targets.argmax(dim=1)

    if world_size > 1 and not local_loss:
        targets = gather_features_single(
            targets, local_loss, gather_with_grad, rank, world_size
        )

    if cur_modal == "rgbd":
        # sim_m2l = (
        #     logit_scale[cur_modal]
        #     * samples_proj2text_features[cur_modal]["rgbd"]
        #     @ modal_labels_features[cur_modal].T
        # )
        if world_size > 1:
            all_image_features, all_text_features = gather_features(
                samples_proj2text_features[cur_modal]["rgbd"],
                modal_labels_features[cur_modal],
                local_loss,
                gather_with_grad,
                rank,
                world_size,
                False,
            )

            if local_loss:
                logits_per_image = (
                    logit_scale[cur_modal]
                    * samples_proj2text_features[cur_modal]["rgbd"]
                    @ all_text_features.T
                )
                # logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale[cur_modal] * all_image_features @ all_text_features.T
                )
                # logits_per_text = logits_per_image.T
        else:
            logits_per_image = (
                logit_scale[cur_modal]
                * samples_proj2text_features[cur_modal]["rgbd"]
                @ modal_labels_features[cur_modal].T
            )

    else:
        if world_size > 1:
            all_image_features, all_text_features = gather_features(
                samples_proj2text_features[cur_modal][cur_modal],
                modal_labels_features[cur_modal],
                local_loss,
                gather_with_grad,
                rank,
                world_size,
                False,
            )

            if local_loss:
                logits_per_image = (
                    logit_scale[cur_modal]
                    * samples_proj2text_features[cur_modal][cur_modal]
                    @ all_text_features.T
                )
                # logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale[cur_modal] * all_image_features @ all_text_features.T
                )
                # logits_per_text = logits_per_image.T
        else:
            logits_per_image = (
                logit_scale[cur_modal]
                * samples_proj2text_features[cur_modal][cur_modal]
                @ modal_labels_features[cur_modal].T
            )

        # sim_m2l = (
        #     logit_scale[cur_modal]
        #     * samples_proj2text_features[cur_modal]
        #     @ modal_labels_features[cur_modal].T
        # )
    m2t_cls_loss = F.cross_entropy(logits_per_image, targets)

    dict_cm_cls_loss = {f"cm_cls_loss_{cur_modal}": m2t_cls_loss}

    m2t_cls_acc = (
        (logits_per_image.detach().softmax(dim=-1).argmax(dim=1) == targets)
        .float()
        .mean()
    )

    dict_cm_cls_acc = {f"cm_cls_acc_{cur_modal}": m2t_cls_acc}

    return dict_cm_cls_loss, dict_cm_cls_acc


def multi_uni_modal_clip_loss_gather(
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    logit_scale,
    epoch=0,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    args=None,
):
    dict_clip_loss = {}

    for modal in modal_list:
        loss = single_uni_modal_clip_loss_gather(
            modal,
            samples_proj2text_features,
            samples_text_features,
            logit_scale,
            epoch,
            local_loss,
            gather_with_grad,
            rank,
            world_size,
            args,
        )

        dict_clip_loss.update(**loss)

    return dict_clip_loss


def multi_cross_modal_kd_loss_gather(
    modal_list,
    samples_proj2text_features,
    samples_text_features,
    modal_labels_features,
    logit_scale,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    dict_kd_loss = {}

    for modal in modal_list:
        loss = single_cross_modal_kd_loss_gather(
            modal,
            modal_list,
            samples_proj2text_features,
            samples_text_features,
            modal_labels_features,
            logit_scale,
            local_loss,
            gather_with_grad,
            rank,
            world_size,
        )

        dict_kd_loss.update(**loss)

    return dict_kd_loss


def multi_cross_modal_cls_gather(
    modal_list,
    samples_proj2text_features,
    modal_labels_features,
    targets,
    logit_scale,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    accum_iter=0,
):
    dict_cm_cls_loss = {}
    dict_cm_cls_acc = {}

    for modal in modal_list:
        loss, acc = cross_modal_cls_gather(
            modal,
            samples_proj2text_features,
            modal_labels_features,
            targets[modal][accum_iter],
            logit_scale,
            local_loss,
            gather_with_grad,
            rank,
            world_size,
        )

        dict_cm_cls_loss.update(**loss)
        dict_cm_cls_acc.update(**acc)

    return dict_cm_cls_loss, dict_cm_cls_acc


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T, logit_stand):
        super(DistillKL, self).__init__()
        self.T = T
        self.logit_stand = logit_stand

    def forward(self, y_s, y_t):
        y_s = kd_normalize(y_s) if self.logit_stand else y_s
        y_t = kd_normalize(y_t) if self.logit_stand else y_t
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)
        return loss


# class KDClipLoss(nn.Module):

#     def __init__(
#             self,
#             args,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size
#         self.use_horovod = use_horovod
#         self.args = args

#         if args.t_embed_dim != args.s_embed_dim:
#             self.visual_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)
#             self.text_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)

#         if args.alpha_afd_loss > 0.:
#             self.visual_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)
#             self.text_fusion_proj = nn.Linear(args.s_embed_dim+args.t_embed_dim, args.s_embed_dim)

#         # cache state
#         self.prev_num_logits = 0
#         self.kl_loss = DistillKL(T=1)
#         self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.labels = {}

#     def forward(self, image_features, text_features, logit_scale, \
#         t_image_features, t_text_features, t_logit_scale):
#         device = image_features.device
#         if self.world_size > 1:
#             all_image_features, all_text_features = gather_features(
#                 image_features, text_features,
#                 self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
#             t_all_image_features, t_all_text_features = gather_features(
#                 t_image_features, t_text_features,
#                 self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

#             t_logits_per_image = t_logit_scale * t_all_image_features @ t_all_text_features.T
#             t_logits_per_text = t_logits_per_image.T

#             normalized_image_features = F.normalize(image_features, dim=1)
#             normalized_text_features = F.normalize(text_features, dim=1)
#             normalized_all_image_features = F.normalize(all_image_features, dim=1)
#             normalized_all_text_features = F.normalize(all_text_features, dim=1)

#             if self.local_loss:
#                 logits_per_image = logit_scale * normalized_image_features @ normalized_all_text_features.T
#                 logits_per_text = logit_scale * normalized_text_features @ normalized_all_image_features.T
#             else:
#                 logits_per_image = logit_scale * normalized_all_image_features @ normalized_all_text_features.T
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = logit_scale * normalized_image_features @ normalized_text_features.T
#             logits_per_text = logit_scale * normalized_text_features @ normalized_image_features.T

#         # calculated ground-truth and cache if enabled
#         num_logits = logits_per_image.shape[0]
#         if self.prev_num_logits != num_logits or device not in self.labels:
#             labels = torch.arange(num_logits, device=device, dtype=torch.long)
#             if self.world_size > 1 and self.local_loss:
#                 labels = labels + num_logits * self.rank
#             if self.cache_labels:
#                 self.labels[device] = labels
#                 self.prev_num_logits = num_logits
#         else:
#             labels = self.labels[device]

#         if self.args.t_embed_dim != self.args.s_embed_dim:
#             all_image_features = self.visual_proj(all_image_features)
#             all_text_features = self.text_proj(all_text_features)

#         normalized_all_image_features = F.normalize(all_image_features, dim=1)
#         normalized_all_text_features = F.normalize(all_text_features, dim=1)
#         fd_loss = F.mse_loss(normalized_all_image_features, t_all_image_features) +\
#             F.mse_loss(normalized_all_text_features, t_all_text_features)

#         logits_per_s_image_to_t_text = self.cross_logit_scale * normalized_all_image_features @ t_all_text_features.T
#         logits_per_s_text_to_t_image = self.cross_logit_scale * normalized_all_text_features @ t_all_image_features.T

#         task_loss = (
#             F.cross_entropy(logits_per_image, labels) +
#             F.cross_entropy(logits_per_text, labels)
#             ) / 2

#         ckd_loss = torch.tensor(0.).cuda()
#         icl_loss = torch.tensor(0.).cuda()
#         cross_kd_loss = torch.tensor(0.).cuda()
#         gd_loss = torch.tensor(0.).cuda()
#         afd_loss = torch.tensor(0.).cuda()

#         icl_loss = (
#             F.cross_entropy(logits_per_s_image_to_t_text, labels) +
#             F.cross_entropy(logits_per_s_text_to_t_image, labels)
#             ) / 2

#         ckd_loss = (self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
#             self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2

#         cross_kd_loss = (self.kl_loss(logits_per_s_image_to_t_text, t_logits_per_image.detach()) +\
#             self.kl_loss(logits_per_s_text_to_t_image, t_logits_per_text.detach())) / 2
#         #kd_loss = (F.cross_entropy(logits_per_image, F.softmax(, dim=1)) \
#         #    + F.cross_entropy(logits_per_text, F.softmax(t_logits_per_text.detach(), dim=1))) / 2


#         if self.args.alpha_gd_loss > 0.:
#             with torch.no_grad():
#                 t_grad_p_img, t_grad_k_txt = get_grad(t_all_image_features, t_all_text_features, t_logit_scale, labels)
#                 t_grad_p_txt, t_grad_k_img = get_grad(t_all_text_features, t_all_image_features, t_logit_scale, labels)

#             s_grad_p_img, s_grad_k_txt = get_grad(normalized_all_image_features, normalized_all_text_features, logit_scale, labels)
#             s_grad_p_txt, s_grad_k_img = get_grad(normalized_all_text_features, normalized_all_image_features, logit_scale, labels)

#             gd_loss = F.mse_loss(s_grad_p_img, t_grad_p_img.detach()) +\
#                 F.mse_loss(s_grad_k_txt, t_grad_k_txt.detach()) +\
#                     F.mse_loss(s_grad_p_txt, t_grad_p_txt.detach()) +\
#                         F.mse_loss(s_grad_k_img, t_grad_k_img.detach())

#         if self.args.alpha_afd_loss > 0.:
#             img_fusion_feat = torch.cat([normalized_all_image_features, t_all_image_features], dim=1)
#             txt_fusion_feat = torch.cat([normalized_all_text_features, t_all_text_features], dim=1)
#             img_fusion_feat = self.visual_fusion_proj(img_fusion_feat)
#             txt_fusion_feat = self.text_fusion_proj(txt_fusion_feat)
#             img_fusion_feat = F.normalize(img_fusion_feat, dim=1)
#             txt_fusion_feat = F.normalize(txt_fusion_feat, dim=1)

#             logits_per_fusion_image = self.fusion_logit_scale * img_fusion_feat @ txt_fusion_feat.T
#             logits_per_fusion_text = logits_per_fusion_image.T
#             afd_loss = (
#                 F.cross_entropy(logits_per_fusion_image, labels) +
#                 F.cross_entropy(logits_per_fusion_text, labels)
#             ) / 2


#         ckd_loss = self.args.alpha_ckd_loss * ckd_loss
#         icl_loss = self.args.alpha_icl_loss * icl_loss
#         cross_kd_loss = self.args.alpha_cross_kd_loss * cross_kd_loss
#         fd_loss = self.args.alpha_fd_loss * fd_loss
#         gd_loss = self.args.alpha_gd_loss * gd_loss
#         afd_loss = self.args.alpha_afd_loss * afd_loss

#         return task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss


class KDClipLoss(nn.Module):
    def __init__(
        self,
        args,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.args = args
        # cache state
        self.prev_num_logits = 0
        # self.kl_loss = DistillKL(T=1)

        self.labels = {}

    def forward(self, image_features, t_image_features):
        if self.world_size > 1:
            image_features = gather_features_single(
                image_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
            )
            t_image_features = gather_features_single(
                t_image_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
            )

        assert (
            image_features.shape == t_image_features.shape
        ), "Shape mismatch between image_features and t_image_features"

        fd_loss = F.mse_loss(image_features, t_image_features)

        # ckd_loss = torch.tensor(0.).cuda()
        # ckd_loss = (self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
        #     self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2

        # ckd_loss = self.args.alpha_ckd_loss * ckd_loss
        fd_loss = self.args.alpha_fd_loss * fd_loss

        return fd_loss


def single_kd_clip_loss_gather(
    cur_modal=None,
    samples_proj2text_features=None,
    samples_text_features=None,
    s_samples_proj2text_features=None,
    t_samples_proj2text_features=None,
    t_text_samples_proj2text_features=None,
    t_image_samples_proj2text_features=None,
    logit_scale=None,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    args=None,
):
    dict_kd_clip_loss = {}

    kd_clip_loss = KDClipLoss(
        args=args,
        local_loss=local_loss,
        gather_with_grad=gather_with_grad,
        cache_labels=True,
        rank=rank,
        world_size=world_size,
    )

    kd_clip_loss = kd_clip_loss(
        s_samples_proj2text_features[cur_modal][cur_modal], t_samples_proj2text_features[cur_modal]
    )
    dict_kd_clip_loss.update({f"kd_fd_loss_{cur_modal}": kd_clip_loss})
    
    return dict_kd_clip_loss

    # if world_size > 1:
        
    #     s_visual_features = gather_features_single(
    #         s_samples_proj2text_features[cur_modal][cur_modal],
    #         local_loss,
    #         gather_with_grad,
    #         rank,
    #         world_size,
    #     )
        
    #     visual_features = gather_features_single(
    #         samples_proj2text_features[cur_modal][cur_modal],
    #         local_loss,
    #         gather_with_grad,
    #         rank,
    #         world_size,
    #     )
        
    #     text_features = gather_features_single(
    #         samples_text_features[cur_modal],
    #         local_loss,
    #         gather_with_grad,
    #         rank,
    #         world_size,
    #     )
        
    #     # if "image" in samples_proj2text_features:
    #     #     image_features = gather_features_single(
    #     #         samples_proj2text_features[cur_modal]["image"],
    #     #         local_loss,
    #     #         gather_with_grad,
    #     #         rank,
    #     #         world_size,
    #     #     )
    #     # else:
    #     #     image_features = None
        
    #     t_visual_features = gather_features_single(
    #         t_samples_proj2text_features[cur_modal],
    #         local_loss,
    #         gather_with_grad,
    #         rank,
    #         world_size,
    #     )
    #     t_text_features = gather_features_single(
    #         t_text_samples_proj2text_features[cur_modal],
    #         local_loss,
    #         gather_with_grad,
    #         rank,
    #         world_size,
    #     )
        
    #     # if cur_modal in t_image_samples_proj2text_features:
    #     #     t_image_features = gather_features_single(
    #     #         t_image_samples_proj2text_features[cur_modal],
    #     #         local_loss,
    #     #         gather_with_grad,
    #     #         rank,
    #     #         world_size,
    #     #     )
    # else:
    #     s_visual_features = s_samples_proj2text_features[cur_modal][cur_modal]
    #     visual_features = samples_proj2text_features[cur_modal][cur_modal]
    #     text_features = samples_text_features[cur_modal]
    #     # if "image" in samples_proj2text_features[cur_modal]:
    #     #     image_features = samples_proj2text_features[cur_modal]["image"]
    #     # else:
    #     #     image_features = None
    #     t_visual_features = t_samples_proj2text_features[cur_modal]
    #     t_text_features = t_text_samples_proj2text_features[cur_modal]
    #     # t_image_features = t_image_samples_proj2text_features[cur_modal]

    # assert (
    #     s_visual_features.shape
    #     == t_visual_features.shape
    #     == t_text_features.shape
    # ), "Shape mismatch between visual_features and t_visual_features"

    # logits_per_vt = logit_scale[cur_modal] * visual_features @ text_features.T
    # logits_per_tv = logits_per_vt.T
    
    # # if image_features is not None:
    # #     logits_per_vi = logit_scale[cur_modal] * visual_features @ image_features.T
    # #     logits_per_iv = logits_per_vi.T
    
    # t_logits_per_vt = logit_scale[cur_modal] * t_visual_features @ t_text_features.T
    # t_logits_per_tv = t_logits_per_vt.T

    # # if t_image_features is not None:
        
    # #     t_logits_per_vi = logit_scale[cur_modal] * t_visual_features @ t_image_features.T
    # #     t_logits_per_iv = t_logits_per_vi.T
    
    # fd_loss = F.mse_loss(s_visual_features, t_visual_features)
    # fd_loss = args.alpha_fd_loss * fd_loss
    # dict_kd_clip_loss.update({f"kd_fd_loss_{cur_modal}": fd_loss})
    
    # ckd_loss = torch.tensor(0.).cuda()
    
    # CKD = DistillKL(T=1.0, logit_stand=True)
    
    # # if image_features is not None:
    # #     ckd_loss = (CKD(logits_per_vt, t_logits_per_vt.detach()) +\
    # #     CKD(logits_per_tv, t_logits_per_tv.detach())) / 2 + (CKD(logits_per_vi, t_logits_per_vi.detach()) +\
    # #     CKD(logits_per_iv, t_logits_per_iv.detach())) / 2
    # # else:
    # ckd_loss = (CKD(logits_per_vt, t_logits_per_vt.detach()) +\
    # CKD(logits_per_tv, t_logits_per_tv.detach())) / 2
    
    # ckd_loss = args.alpha_ckd_loss * ckd_loss
    # dict_kd_clip_loss.update({f"kd_ckd_loss_{cur_modal}": ckd_loss})
    
    
    # return dict_kd_clip_loss
