from enum import Enum

from src.explanation.wrappers.shap_wrapper import shap_explain
# from src.explanation.wrappers.lrp_wrapper import lrp_explain


class ExplainerType(Enum):
    SHAP = 'shap'


def explain(explainer, predictive_model, test_df, encoder):
    if explainer is ExplainerType.SHAP.value:
        return shap_explain(predictive_model, test_df, encoder)
    else:
        raise Exception('selected explainer not yet supported')
