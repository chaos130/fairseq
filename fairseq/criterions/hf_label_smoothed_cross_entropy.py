# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class HFLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "hf_label_smoothed_cross_entropy", dataclass=HFLabelSmoothedCrossEntropyCriterionConfig
)
class HFLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        #print(f'net_output:{net_output}')
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        rrhf_loss = self.compute_rrhf_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "rrhf_loss": utils.item(rrhf_loss.data) if reduce else rrhf_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        loss += rrhf_loss
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    def compute_normalized_probs(self, logits, log_probs):
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=False)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=False)

    def compute_sentence_probs(self, logits, target, ignore_index=None, length_penalty=1.0):

        lprobs = self.compute_normalized_probs(logits, log_probs=True)

        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        #print(f'probs:{lprobs}, {lprobs.size()}')
        #print(f'target:{target}, {target.size()}')
        sent_probs = lprobs.gather(dim=-1, index=target)
        #print(f'sent_probs:{sent_probs}, {sent_probs.size()}')

        sent_length = None
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            # print(f'pad_mask:{pad_mask}, {pad_mask.size()}')

            sent_probs.masked_fill_(pad_mask, 0.0)
            #print(f'sent_probs:{sent_probs}, {sent_probs.size()}')
            sent_length = torch.sum(~pad_mask.squeeze(-1), dim=-1)
        
        sent_probs = sent_probs.squeeze(-1).sum(dim=-1)

        if sent_length is not None:
            sent_probs = sent_probs / sent_length ** length_penalty
        #rint(f'sent_probs:{sent_probs}, {sent_probs.size()}')

        return sent_probs

    def compute_rrhf_loss(self, model, net_output, sample, reduce=True):
        hf_logit_list = net_output[2]
        hf_num = len(hf_logit_list)
        hf_score_list = sample['hf_score_list']
        hf_target_list = sample['hf_target_list']


        # print(f'hf_logit_list:{hf_logit_list}')
        # print(f'hf_target_list:{hf_target_list}')
        # print(f'hf_score_list: {hf_score_list}')
        assert len(hf_score_list) == hf_num, (len(hf_score_list), hf_num)

        sent_prob_list = []
        for hf_logit, hf_target in zip(hf_logit_list, hf_target_list):
            sent_prob = self.compute_sentence_probs(hf_logit, hf_target, self.padding_idx)
            sent_prob_list.append(sent_prob)

        model_scores = torch.stack(sent_prob_list, 0)
        hf_scores = torch.stack(hf_score_list, 0)

        # print(f'model_score:{model_scores}')
        # print(f'hf_score:{hf_score}')

        model_scores = model_scores.transpose(0, 1)
        hf_scores = hf_scores.transpose(0, 1)

        # print(f'model_score:{model_scores}') # bs * hf_num
        # print(f'hf_score:{hf_scores}') # bs * hf_num

        rrhf_loss = 0.0
        for i in range(model_scores.shape[0]):
            model_diff = model_scores[i].unsqueeze(0) - model_scores[i].unsqueeze(-1)
            hf_diff = hf_scores[i].unsqueeze(0) - hf_scores[i].unsqueeze(-1)
            aval = torch.bitwise_and(hf_diff>0, model_diff<0)
            rrhf_loss += -model_diff[aval].sum().mean()
        # print(f'rrhf_loss: {rrhf_loss}')

        return rrhf_loss 

    def compute_dpo_loss(self, model, net_output, sample, reduce=True, beta=0.1):
        hf_logit_list = net_output[2]
        hf_num = len(hf_logit_list)
        hf_score_list = sample['hf_score_list'] # the log_prob caculated by the SFT model
        hf_target_list = sample['hf_target_list']
        assert hf_num == 2, hf_num

        sent_prob_list = []
        for hf_logit, hf_target in zip(hf_logit_list, hf_target_list):
            sent_prob = self.compute_sentence_prob(hf_logit, hf_target, self.padding_idx)
            sent_prob_list.append(sent_prob)

        model_scores = torch.stack(sent_prob_list, 0)
        hf_scores = torch.stack(hf_score_list, 0)

        model_scores = model_scores.transpose(0, 1)
        hf_scores = hf_scores.transpose(0, 1)

        dpo_loss = 0.0
        for i in range(model_scores.shape[0]):
            pi_logratios = model_scores[i][0] - model_scores[i][1]
            ref_logratios = hf_scores[i][0] - hf_scores[i][0]
            dpo_loss += -F.logsigmoid(beta * (pi_logratios - ref_logratios))

        return dpo_loss


    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
