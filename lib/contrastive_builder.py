from torch import distributed as dist
from torch import nn

from lib.model import get_model_class, HeadWrapper, CepLossBuilder


class ModelFactory:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def build_normal(self):
        base_model_class = get_model_class(self.cfg.arch)
        num_classes = self.cfg.finetune_data.dataset.num_classes

        model = HeadWrapper(
            base_model_class,
            logger=self.logger,
            num_classes=num_classes,  # only use when in finetuning mode
            finetune=True,
        )

        model = self._normal_post_process_model(model)

        model.cuda()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            find_unused_parameters=True,
        )
        return model

    def _normal_post_process_model(self, model: nn.Module):
        # finetuning all or linear probe
        # default finetuning all

        if self.cfg.get('only_train_fc', False):
            for param in model.parameters():
                param.requires_grad = False

            fc_names = ['fc', 'new_fc']
            fc_module = next(getattr(model, n) for n in fc_names if hasattr(model, n))
            if fc_module is None:
                raise Exception('"only_train_fc" specified, but no fc layer found')

            for param in fc_module.parameters():
                param.requires_grad = True

            orig_train = model.train

            def override_train(mode=True):
                orig_train(mode=False)
                fc_module.train(mode)

            model.train = override_train

            self.logger.info('In linear probe mode only fc layers will be trained')

        return model

    def build_cep_with_contrastive_loss(self):
        dim = self.cfg.contrastive.dim  # op dim
        t = self.cfg.contrastive.t  # temperature
        k = self.cfg.contrastive.k  # cache queue length
        m = self.cfg.contrastive.m  # momentum para
        fc_type = self.cfg.contrastive.fc_type
        projectorbn = self.cfg.contrastive.get('projectorbn', [False, True])

        base_model_class = get_model_class(self.cfg.arch)

        def model_class(num_classes=128):
            model = HeadWrapper(
                base_model_class,
                logger=self.logger,
                num_classes=num_classes,  # only use when in finetuning mode
                fc_type=fc_type,
                finetune=False,
                groups=1,
                projectorbn=projectorbn)
            return model

        model = CepLossBuilder(model_class,
                               logger=self.logger,
                               dim=dim,
                               K=k,
                               m=m,
                               T=t,
                               predictor_layer=[4, 4],
                               predictor_meddim=None,
                               predictor_usebn=True)

        model.cuda()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            find_unused_parameters=True,
        )

        return model


# if __name__ == "__main__":
#
#     from mmcv import Config
#     import logging
#     import torch
#     from model import Loss
#     cfg = Config.fromfile('/home/youshyee/space/codespace/cyclepre/configs/samplecfg.py')
#
#     logger = logging.getLogger(__name__)
#     M = ModelFactory(cfg, logger)
#     criterion = Loss()
#     model = M.build_cep_with_contrastive_loss()
#     clip_q, clip_k = torch.randn(2, 3, 16, 64, 64), torch.randn(2, 3, 16, 64, 64)
#     logits_A, mask_A, logits_C_1, logits_C_2, CC_1, CC_2 = model(clip_q, clip_k)
#     loss, Aloss, CLRankLoss, CC_dis = criterion(
#         Alogits=logits_A,
#         mask=mask_A,
#         ranking_logits1=logits_C_1,
#         ranking_logits2=logits_C_2,
#         CC_1=CC_1,
#         CC_2=CC_2,
#     )
#     print(loss, Aloss, CLRankLoss, CC_dis)
#     loss.backward()
