import pandas as pd

baseline_query = 'pred_type == "mixture" and model == "no_regul"'
old_regul_queries = {
    'mixture_cdf_based (L=1)': 'pred_type == "mixture" and model == "cdf_based" and L == 1',
    'mixture_cdf_based (L=2)': 'pred_type == "mixture" and model == "cdf_based" and L == 2',
    'mixture_quantile_based (L=1)': 'pred_type == "mixture" and model == "quantile_based" and L == 1 and neural_sort == True',
    'mixture_quantile_based (L=2)': 'pred_type == "mixture" and model == "quantile_based" and L == 2 and neural_sort == True',
    'mixture_entropy_based': 'pred_type == "mixture" and model == "entropy_based" and neural_sort == True',
    'mixture_ic': 'pred_type == "mixture" and model == "ic"',
    'mixture_oqr': 'pred_type == "mixture" and model == "oqr"',
    'quantile_quantile_based (L=1)': 'pred_type == "quantile" and model == "quantile_based" and L == 1 and n_quantiles == 1',
    'quantile_quantile_based (L=2)': 'pred_type == "quantile" and model == "quantile_based" and L == 2 and n_quantiles == 1',
}

misspec_queries = {
    'small_mlp': 'misspecification == "small_mlp" and misspecification.notna()',
    'big_mlp': 'misspecification == "big_mlp" and misspecification.notna()',
    'homoscedasticity': 'misspecification == "homoscedasticity" and misspecification.notna()',
    'sharpness_reward': 'misspecification == "sharpness_reward" and misspecification.notna()',
    'mixture_size_1': 'mixture_size == 1 and mixture_size.notna()',
    'drop_prob_0_9': 'drop_prob == 0.9 and drop_prob.notna()',
}
misspec_queries = {key: f'({value})' for key, value in misspec_queries.items()}

any_misspec_query = ' or '.join(misspec_queries.values())
any_misspec_query = f'({any_misspec_query})'
no_misspec_query = f'(not {any_misspec_query})'

all_misspec_queries = {f'misspec_{name}': query for name, query in misspec_queries.items()}
all_misspec_queries['no_misspec'] = no_misspec_query

model_names = {
    'no_regul': None,
    'entropy_based': 'QR',
    'cdf_based': 'PCE-KDE',
    'quantile_based': 'PCE-Sort',
    'truncated': 'Trunc',
    'oqr': 'OQR',
    'ic': 'IC',
}
posthoc_names = {
    'ecdf': 'Recal',
    'smooth_ecdf': 'KDE Recal',
    'stochastic_ecdf': 'Stochastic Recal',
    'linear_ecdf': 'Linear Recal',
    'CQR': 'CQR',
}
metric_names = {
    'nll': 'NLL',
    'crps': 'CRPS',
    'wis': 'CRPS',
    'calib_l1': 'PCE',
    'calib_l2': 'PCE_2',
    'stddev': 'STD',
    'mae': 'MAE',
    'rmse': 'RMSE',
}
base_model_names = {
    'nll': 'MIX-NLL',
    'crps': 'MIX-CRPS',
    'expected_qs': 'SQR-CRPS',
}


def model_name(d, add_posthoc_dataset=True):
    base_loss, regul = (
        base_model_names[d['base_loss']],
        model_names[d['model']],
    )
    name = base_loss
    if regul is not None:
        name += f' + {regul}'
    if 'posthoc_method' in d and not pd.isna(d['posthoc_method']):
        method = d['posthoc_method']
        if method in posthoc_names:
            method = posthoc_names[method]
        name += f' + {method}'
        if add_posthoc_dataset:
            posthoc_dataset = d['posthoc_dataset']
            name += f' ({posthoc_dataset})'
    return name
