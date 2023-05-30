import collections

from omegaconf import OmegaConf

from uq.models.dist.base_dist_module import DistModule
from uq.models.dist.ic_module import DistIC_Regul
from uq.models.dist.marginal_regul_module import DistCDF_Regul, DistEntropyRegul, DistQuantileRegul
from uq.models.dist.oqr_module import DistOQR_Regul
from uq.models.dist.recalibration_module import DistRecalibrated
from uq.models.quantile.base_quantile_module import QuantileModule
from uq.models.quantile.marginal_regul_module import (
    HQPIRegul,
    QuantileCDFRegul,
    TruncatedDistRegul,
)
from uq.models.quantile.oqr_module import QuantileOQR_Regul
from uq.models.quantile.recalibration_module import QuantileRecalibrated
from uq.utils.hparams import Choice, Join, Union


def models_cls_config(config):
    models_cls = dict(
        quantile_no_regul=dict(cls=QuantileModule, trainer='lightning', args={}),
        quantile_cdf_based=dict(cls=QuantileCDFRegul, trainer='lightning', args={}),
        quantile_truncated=dict(cls=TruncatedDistRegul, trainer='lightning', args={}),
        quantile_hqpi=dict(cls=HQPIRegul, trainer='lightning', args={}),
        quantile_oqr=dict(cls=QuantileOQR_Regul, trainer='lightning', args={}),
        quantile_recalibrated=dict(cls=QuantileRecalibrated, trainer='lightning', args={}),
        dist_no_regul=dict(cls=DistModule, trainer='lightning', args={}),
        dist_oqr=dict(cls=DistOQR_Regul, trainer='lightning', args={}),
        dist_ic=dict(cls=DistIC_Regul, trainer='lightning', args={}),
        dist_entropy_based=dict(cls=DistEntropyRegul, trainer='lightning', args={}),
        dist_cdf_based=dict(cls=DistCDF_Regul, trainer='lightning', args={}),
        dist_quantile_based=dict(cls=DistQuantileRegul, trainer='lightning', args={}),
        dist_recalibrated=dict(cls=DistRecalibrated, trainer='lightning', args={}),
    )
    return OmegaConf.create(dict(models_cls=models_cls), flags={'allow_objects': True})


def get_hparam_tuning(config):
    return {
        **get_tuning_for_no_regul(config),
        **get_tuning_for_regul(config),
    }


def get_tuning_for_no_regul(config):
    mlp = Join(
        Choice(nb_hidden=[1, 2, 3, 5, 10, 20]),
    )
    dist_no_regul = mlp(
        Choice(pred_type='mixture'),
        Choice(base_loss=['nll', 'crps']),
        Choice(mixture_size=[1, 3, 5, 10, 20, 50]),
        Choice(model='dist_no_regul'),
    )
    quantile_no_regul = mlp(
        Choice(pred_type='quantile'),
        Choice(base_loss='expected_qs'),
        Choice(n_quantiles=[1, 4, 16, 64]),
        Choice(model='quantile_no_regul'),
    )
    models_tuning = dict(
        no_regul=Union(dist_no_regul, quantile_no_regul),
    )
    return models_tuning


def get_tuning_for_regul(config):
    lambda_ = Choice(lambda_=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
    L = Choice(L=[1])

    mlp = Join(
        Choice(nb_hidden=3),
    )
    dist = Join(
        Choice(pred_type='mixture'),
        Choice(mixture_size=3),
        Choice(base_loss=['nll', 'crps']),
    )
    quantile = Join(
        Choice(pred_type='quantile'),
        Choice(n_quantiles=64),
        Choice(base_loss=['expected_qs']),
    )
    dist_regul = dist(lambda_)
    quantile_regul = quantile(lambda_)

    cdf_based = Union(
        dist_regul(Choice(model='dist_cdf_based'), Choice(s=[10, 20, 50, 100, 200]), L),
        quantile_regul(
            Choice(model='quantile_cdf_based'),
            Choice(s=[10, 20, 50, 100, 200]),
            L,
        ),
    )
    quantile_based = dist_regul(
        Choice(model='dist_quantile_based'),
        Choice(neural_sort=True),
        Choice(s=[0.001, 0.01, 0.1, 1]),
        L,
    )
    entropy_based = dist_regul(
        Choice(model='dist_entropy_based'),
        Choice(neural_sort=True),
        Choice(s=[0.001, 0.01, 0.1, 1]),
    )

    models_tuning = dict(
        cdf_based=cdf_based,
        quantile_based=quantile_based,
        entropy_based=entropy_based,
    )
    return models_tuning


def get_tuning_for_layers(config):
    nb_hidden = Choice(nb_hidden=[1, 3, 5, 10, 20, 50, 100])

    mlp = Join(
        nb_hidden,
    )
    dist = mlp(
        Choice(pred_type='mixture'),
        Choice(model='dist_no_regul'),
        Choice(base_loss=['nll']),
    )
    models_tuning = dict(
        no_regul=dist,
    )
    return models_tuning


def get_tuning_for_mixture_nll(config):
    mlp = Join(
        Choice(nb_hidden=[3]),
    )
    dist = mlp(
        Join(
            Choice(pred_type='mixture'),
            Choice(mixture_size=[5]),
            Choice(base_loss=['nll']),
        ),
    )
    no_regul = dist(Choice(model='dist_no_regul'))

    models_tuning = dict(
        no_regul=no_regul,
    )
    return models_tuning


def get_tuning_for_all(
    config,
    lambda_=None,
    misspecifications=True,
    enable_interleaved=False,
    use_spline=False,
    regul=True,
    posthoc=True,
):

    if lambda_ is None:
        lambda_ = [0.2, 1, 5]
    nb_hidden = Choice(nb_hidden=[3])
    mixture_size = Choice(mixture_size=[3])
    count_bins = Choice(count_bins=[8])
    n_quantiles = Choice(n_quantiles=[64])
    lambda_ = Choice(lambda_=lambda_)
    interleaved = Choice(interleaved=[False])
    if enable_interleaved:
        interleaved.values.append(True)
    neural_sort = Choice(neural_sort=[True])
    L = Choice(L=[1])
    s = Choice(s=[100.0])   # 50.
    posthoc_dataset = Choice(posthoc_dataset=['calib', 'train'])

    if misspecifications:
        misspecification_mixture = Union(
            Choice(
                misspecification=[
                    None,
                    # 'small_mlp',
                    # 'big_mlp',
                    # 'homoscedasticity',
                    # 'sharpness_reward',
                ]
            ),
            Choice(mixture_size=1),
        )
        misspecification_quantile = Union(
            Choice(
                misspecification=[
                    None,
                    # 'small_mlp',
                    # 'big_mlp',
                ]
            ),
        )
    else:
        misspecification_mixture = Choice(misspecification=None)
        misspecification_quantile = Choice(misspecification=None)

    posthoc_mlp = Join(
        posthoc_dataset,
    )
    posthoc_dist = Union(
        Join(),
    )
    posthoc_quantile = Union(
        Join(),
    )
    if posthoc:
        posthoc_dist = posthoc_dist(
            posthoc_mlp(Choice(posthoc_method=['ecdf', 'linear_ecdf', 'smooth_ecdf']))
        )
        posthoc_quantile = posthoc_quantile(posthoc_mlp(Choice(posthoc_method=['CQR'])))

    mlp = Join(
        nb_hidden,
    )
    mixture = Join(
        Choice(pred_type='mixture'),
        mixture_size,
        Choice(base_loss=['nll', 'crps']),
        misspecification_mixture,
    )
    if use_spline:
        spline = Join(Choice(pred_type='spline'), count_bins, Choice(base_loss='nll'))
        dist = mlp(Union(mixture, spline))
    else:
        dist = mlp(mixture)

    quantile = mlp(
        Choice(pred_type='quantile'),
        n_quantiles,
        Choice(base_loss='expected_qs'),
        misspecification_quantile,
    )

    no_regul = Union(
        dist(Choice(model='dist_no_regul'), posthoc_dist),
        quantile(Choice(model='quantile_no_regul'), posthoc_quantile),
    )

    if regul:
        dist_regul = dist(lambda_, interleaved)
        quantile_regul = quantile(lambda_, interleaved)
        cdf_based = Union(
            dist_regul(Choice(model='dist_cdf_based'), s, L),
            quantile_regul(Choice(model='quantile_cdf_based'), s, L),
        )
        quantile_based = dist_regul(Choice(model='dist_quantile_based'), neural_sort, L)
        entropy_based = dist_regul(Choice(model='dist_entropy_based'), neural_sort)
        truncated = quantile_regul(Choice(model='quantile_truncated'))

        models_tuning = dict(
            no_regul=no_regul,
            cdf_based=cdf_based,
            quantile_based=quantile_based,
            entropy_based=entropy_based,
            truncated=truncated,
        )
    else:
        models_tuning = dict(
            no_regul=no_regul,
        )

    return models_tuning


def _get_tuning(config):
    if config.tuning_type == 'all':
        return get_tuning_for_all(config, enable_interleaved=True, use_spline=True)
    elif config.tuning_type == 'misspec_and_regul_and_posthoc':
        return get_tuning_for_all(config)
    elif config.tuning_type == 'regul_and_posthoc':
        return get_tuning_for_all(config, misspecifications=False, lambda_=[0.01, 0.05, 0.2, 1, 5])
    elif config.tuning_type == 'posthoc':
        return get_tuning_for_all(config, misspecifications=False, regul=False)
    elif config.tuning_type == 'interleaving':
        return get_tuning_for_all(config, enable_interleaved=True)
    elif config.tuning_type == 'mixture_nll':
        return get_tuning_for_mixture_nll(config)
    elif config.tuning_type == 'layers':
        return get_tuning_for_layers(config)
    elif config.tuning_type == 'hparam_tuning':
        return get_hparam_tuning(config)
    raise ValueError('Invalid tuning type')


def duplicates(choices):
    frozendict = lambda d: frozenset(d.items())
    frozen_choices = map(frozendict, choices)
    return [choice for choice, count in collections.Counter(frozen_choices).items() if count > 1]


def get_tuning(config):
    tuning = _get_tuning(config)
    for model, choices in tuning.items():
        dup = duplicates(choices)
        assert len(dup) == 0, dup
    return tuning
