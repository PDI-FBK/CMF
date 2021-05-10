from csv import DictWriter

from scripts.run_full_pipeline import run_full_pipeline
from src.encoding.common import EncodingType, EncodingTypeAttribute
from src.explanation.common import ExplainerType
from src.hyperparameter_optimisation.common import HyperoptTarget
from src.labeling.common import LabelTypes
from src.predictive_model.common import PredictionMethods

results = {}


def prepare_results_for_print(CONF, result):
    ready_for_print = dict()

    # CONF -- data
    ready_for_print['train_set'] = CONF['data']['TRAIN_DATA']
    ready_for_print['validation_set'] = CONF['data']['VALIDATE_DATA']
    ready_for_print['feedback_set'] = CONF['data']['FEEDBACK_DATA']
    ready_for_print['test_set'] = CONF['data']['TEST_DATA']
    ready_for_print['dataset'] = CONF['data']['dataset']
    ready_for_print['formula'] = CONF['data']['formula']

    # CONF -- params
    ready_for_print['prefix_length'] = CONF['prefix_length']
    ready_for_print['padding'] = CONF['padding']
    ready_for_print['feature_selection'] = CONF['feature_selection']
    ready_for_print['labeling_type'] = CONF['labeling_type']
    ready_for_print['predictive_model'] = CONF['predictive_model']
    ready_for_print['explanator'] = CONF['explanator']
    ready_for_print['threshold'] = CONF['threshold']
    ready_for_print['top_k'] = CONF['top_k']
    ready_for_print['hyperparameter_optimisation'] = CONF['hyperparameter_optimisation']
    ready_for_print['hyperparameter_optimisation_epochs'] = CONF['hyperparameter_optimisation_epochs']

    # FEEDBACK
    ready_for_print['feedback_1'] = str(result['used_feedback'][1])
    ready_for_print['feedback_2'] = str(result['used_feedback'][2])
    ready_for_print['feedback_10'] = str(result['feedback_10'])

    # RESULT -- initial
    ready_for_print['initial_result_auc'] = result['initial_result']['auc']
    ready_for_print['initial_result_f1_score'] = result['initial_result']['f1_score']
    ready_for_print['initial_result_accuracy'] = result['initial_result']['accuracy']
    ready_for_print['initial_result_precision'] = result['initial_result']['precision']
    ready_for_print['initial_result_recall'] = result['initial_result']['recall']

    def get_avg_min_max_stdev(k, result):
        avg_min_max_stdev = {}
        # RESULT -- retrain 1 - avg
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_auc_avg'] = result['avg']['auc']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_f1_score_avg'] = result['avg']['f1_score']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_accuracy_avg'] = result['avg']['accuracy']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_precision_avg'] = result['avg']['precision']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_recall_avg'] = result['avg']['recall']
        # RESULT -- retrain 1 - min
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_auc_min'] = result['min']['auc']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_f1_score_min'] = result['min']['f1_score']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_accuracy_min'] = result['min']['accuracy']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_precision_min'] = result['min']['precision']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_recall_min'] = result['min']['recall']
        # RESULT -- retrain 1 - max
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_auc_max'] = result['max']['auc']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_f1_score_max'] = result['max']['f1_score']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_accuracy_max'] = result['max']['accuracy']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_precision_max'] = result['max']['precision']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_recall_max'] = result['max']['recall']
        # RESULT -- retrain 1 - stdev
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_auc_stdev'] = result['stdev']['auc']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_f1_score_stdev'] = result['stdev']['f1_score']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_accuracy_stdev'] = result['stdev']['accuracy']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_precision_stdev'] = result['stdev']['precision']
        avg_min_max_stdev['top_'+str(k)+'_retrain_result_recall_stdev'] = result['stdev']['recall']
        return avg_min_max_stdev

    ready_for_print.update(get_avg_min_max_stdev(1, result['retrain_result'][1]))
    ready_for_print.update(get_avg_min_max_stdev(2, result['retrain_result'][2]))

    return ready_for_print


for dataset, formula, prefix, encoding, train_set, validation_set, feedback_set, test_set in [

    ('synthetic', 'relabelling_7c/', 5, EncodingType.COMPLEX.value, '0-32', '32-40', '40-50', '50-60'),
    ('synthetic', 'relabelling_8c/', 5, EncodingType.COMPLEX.value, '0-32', '32-40', '40-50', '50-60'),
    ('synthetic', 'relabelling_9c/', 5, EncodingType.COMPLEX.value, '0-32', '32-40', '40-50', '50-60'),

    ]:
    CONF = {  # This contains the configuration for the run
        'data':
            {
                'TRAIN_DATA': 'input_data/remake_forum_experiments/' + dataset + '/' + formula + train_set + '.xes',
                'VALIDATE_DATA': 'input_data/remake_forum_experiments/' + dataset + '/' + formula + validation_set + '.xes',
                'FEEDBACK_DATA': 'input_data/remake_forum_experiments/' + dataset + '/' + formula + feedback_set + '.xes',
                'TEST_DATA': 'input_data/remake_forum_experiments/' + dataset + '/' + formula + test_set + '.xes',
                'OUTPUT_DATA': 'output_data/output_data.csv',
                'dataset': dataset,
                'formula': formula
            },
        'prefix_length': prefix,
        'padding': True,
        'feature_selection': encoding,
        'attribute_encoding': EncodingTypeAttribute.ONEHOT.value,  # LABEL, ONEHOT
        'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
        'predictive_model': PredictionMethods.LSTM.value,
        'explanator': ExplainerType.LRP.value,
        'threshold': 13,
        'top_k': 10,
        'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
        'hyperparameter_optimisation': True,
        'hyperparameter_optimisation_epochs': 3,
        'min_supp_miner': 2
    }

    result = run_full_pipeline(CONF)
    print(formula)
    print(result)

    with open(CONF['data']['OUTPUT_DATA'], 'a+') as output_file:
        ready_for_print = prepare_results_for_print(CONF, result)

        dict_writer = DictWriter(output_file, fieldnames=list(ready_for_print.keys()))
        dict_writer.writerow(ready_for_print)
        output_file.close()

results.update({str(CONF): result})

print('##############################################################################################################')
print('##############################################################################################################')
print('')
print(results)
print('')
print('##############################################################################################################')
print('##############################################################################################################')

