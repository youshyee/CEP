from torch import distributed as dist
from torch import nn
import torch

from .model import get_model_class, HeadWrapper, CepLossBuilder, SyncTGN


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
        if self.cfg.get('only_train_fc', False):
            for param in model.parameters():
                param.requires_grad = False

            fc_names = ['fc', 'new_fc']
            fc_module = next(getattr(model, n)
                             for n in fc_names if hasattr(model, n))
            if fc_module is None:
                raise Exception(
                    '"only_train_fc" specified, but no fc layer found')

            for param in fc_module.parameters():
                param.requires_grad = True

            orig_train = model.train

            def override_train(mode=True):
                orig_train(mode=False)
                fc_module.train(mode)

            model.train = override_train

            self.logger.info(
                'In linear probe mode only fc layers will be trained')

        return model

    def tgnwapper(self, module, segment=3):
        module_output = module
        if isinstance(module, nn.modules.batchnorm.BatchNorm3d):
            module_output = SyncTGN(
                segment,
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, self.tgnwapper(child, segment)
            )
        del module
        return module_output

    def build_cep_with_contrastive_loss(self):

        dim = 2048  # op dim
        t = 0.07  # temperature
        k = 16384
        m = 0.99
        fc_type = 'mlp'
        projectorbn = [False, True]

        arch = self.cfg.arch
        base_model_class = get_model_class(arch)

        def model_class(num_classes=dim):
            model = HeadWrapper(
                base_model_class,
                logger=self.logger,
                num_classes=num_classes,  # only use when in finetuning mode
                fc_type=fc_type,
                finetune=False,
                groups=1,
                projectorbn=projectorbn,
                pool=False if arch == 'slowfast' else True)
            return model

        model = CepLossBuilder(model_class,
                               logger=self.logger,
                               dim=dim,
                               K=k,
                               m=m,
                               T=t,
                               predictor_layer=[2, 2],
                               predictor_meddim=4096,
                               predictor_usebn=True)

        model.cuda()
        # TGN wapper
        self.tgnwapper(model, segment=2)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()

        gpu_nums = torch.cuda.device_count()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank() % gpu_nums],
            find_unused_parameters=True,
        )

        return model
