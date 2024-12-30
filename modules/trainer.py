import os
from collections import defaultdict
import random
from pprint import PrettyPrinter

import tensorflow as tf
import numpy as np
import language_evaluation
from tqdm import tqdm
import colorful
from sklearn.metrics import accuracy_score

from data import vocabulary as data_vocab
from data.wizard_of_wikipedia import (
    WowDatasetReader, PARLAI_KNOWLEDGE_SEPARATOR, BERT_KNOWLEDGE_SEPARATOR
)
from utils.etc_utils import check_none_gradients, check_nan_gradients
from models import BaseModel
from modules.from_parlai import normalize_answer
from utils.my_print import my_print

pformat = PrettyPrinter().pformat


class Trainer(object):
    def __init__(self,
                 model: BaseModel,
                 optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 mirrored_strategy: tf.distribute.Strategy = None,
                 enable_function: bool = True,
                 preprocess_fn = lambda x: x):
        self.model = model
        self.optimizer = optimizer
        self.mirrored_strategy = mirrored_strategy
        self.enable_function = enable_function
        self.preprocess_fn = preprocess_fn

        self._batch_size = model.hparams.batch_size
        self._num_gpus = model.hparams.num_gpus

        # Graph上的无监督学习
        self.uns_train_step_fn = self._get_unsupervise_train_step_fn()# single or multi card

        self.train_step_fn = self._get_train_step_fn()# single or multi card
        self.test_step_fn = self._get_test_step_fn()# single or multi card

        # tensorflow2 静态图加速
        if self.enable_function:
            self.uns_train_step_fn = tf.function(self.uns_train_step_fn)
            self.train_step_fn = tf.function(self.train_step_fn)
            self.test_step_fn = tf.function(self.test_step_fn)

    def unsupervise_train_step(self, example):
        return self.uns_train_step_fn(self.model, self.optimizer, example)

    def train_step(self, example):
        return self.train_step_fn(self.model, self.optimizer, example)

    def test_step(self, example):
        return self.test_step_fn(self.model, example)

    # XXX
    def ugly_load_by_single_train_step(self, example):
        example = self.preprocess_fn(example)
        output_dict = self.model(example)
        return output_dict

    def test_loop(self, dataset, num_steps, epoch, mode):# single or multi card
        results_dict = defaultdict(list)
        dataset_iter = iter(dataset)
        test_tqdm = tqdm(range(num_steps), ncols=70, desc=f"Epoch {epoch} (test {mode})")
        for i, current_step in enumerate(test_tqdm):
            example = next(dataset_iter)
            step_result = self.test_step(example)
            for key, value in step_result.items():
                results_dict[key].append(value.numpy())

        for key, value in results_dict.items():
            if key == 'knowledge_sent_gt':# 变长list of knowledge sentences
                list_of_know_sent = [j for i in value for j in i]
                list_of_know_sent = [np.pad(i, ((0,0),(1,0))) for i in list_of_know_sent]
                results_dict[key] = [np.concatenate(i) for i in list_of_know_sent]
            elif key == 'knowledge_sent_pred':# 变长knowledge sentences
                results_dict[key] = [j for i in value for j in i]
            elif results_dict[key][0].shape == ():
                results_dict[key] = np.array(value)
            else:
                results_dict[key] = np.concatenate(value, axis=0)
                    

        return results_dict

    def _get_unsupervise_train_step_fn(self):# single or multi card
        def _train_step(model, optimizer, example):
            # example = self.preprocess_fn(example)
            with tf.GradientTape() as tape:
                output_dict = model.compute_global_loss(example)
                # # XXX : need to exactly compute loss per gpu
                # batch_size = output_dict['num_valid_turns'] if 'num_valid_turns' \
                #     in output_dict else self._batch_size
                # for key, value in output_dict.items():
                #     if 'loss' in key:
                #         output_dict[key] = tf.reduce_sum(value) * 1. / (batch_size * self._num_gpus)
                grads = tape.gradient(output_dict['loss'], model.trainable_variables)
                check_none_gradients(grads, model.trainable_variables, model.hparams.ignore_none_gradients)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return output_dict

        # def _dist_train_step(model, optimizer, example):
        #     with self.mirrored_strategy.scope():
        #         output_dict = self.mirrored_strategy.experimental_run_v2(_train_step, (model, optimizer, example))
        #         for key, value in output_dict.items():
        #             if 'loss' in key:
        #                 output_dict[key] = self.mirrored_strategy.reduce(
        #                     tf.distribute.ReduceOp.SUM, value, axis=None)
        #             else:
        #                 value = self.mirrored_strategy.experimental_local_results(value)
        #                 if key == 'num_valid_turns':  # XXX : change this condition to check whether it is scalar
        #                     value = tf.stack(value, axis=0)
        #                 else:
        #                     value = tf.concat(value, axis=0)
        #                 output_dict[key] = value
        #     return output_dict

        # return _dist_train_step if self.mirrored_strategy else _train_step
        return _train_step

    def _get_train_step_fn(self):# single or multi card
        def _train_step(model, optimizer, example):
            example = self.preprocess_fn(example)
            with tf.GradientTape() as tape:
                output_dict = model(example)
                # XXX : need to exactly compute loss per gpu
                batch_size = output_dict['num_valid_turns'] if 'num_valid_turns' \
                    in output_dict else self._batch_size
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = tf.reduce_sum(value) * 1. / (batch_size * self._num_gpus)
                grads = tape.gradient(output_dict['loss'], model.trainable_variables)
                check_none_gradients(grads, model.trainable_variables, model.hparams.ignore_none_gradients)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return output_dict

        def _dist_train_step(model, optimizer, example):
            with self.mirrored_strategy.scope():
                output_dict = self.mirrored_strategy.experimental_run_v2(_train_step, (model, optimizer, example))
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = self.mirrored_strategy.reduce(
                            tf.distribute.ReduceOp.SUM, value, axis=None)
                    else:
                        value = self.mirrored_strategy.experimental_local_results(value)
                        if key == 'num_valid_turns':  # XXX : change this condition to check whether it is scalar
                            value = tf.stack(value, axis=0)
                        else:
                            value = tf.concat(value, axis=0)
                        output_dict[key] = value
            return output_dict

        return _dist_train_step if self.mirrored_strategy else _train_step


    def _update_train_step_fn_for_student(self):
        # TODO: add params filter; FIXME: I guess we need to pull it in the _train_step
        # [bert_model[TransformerEncoder], embedding_and_softmax, ctx_modeling, posterior_knowledge_selection, prior_knowledge_selection, TransformerDecoder]
        # prior_list=["prior_knowledge_selection"]

        rm_list = ["bert_model","dialog_rnn","history_rnn","posterior_query_layer","predict"]
        def get_vlist(vlist, rm_list=None, vlist_print=False):
            # pdb.set_trace()

            # save_list = ["prior_knowledge_selection","TransformerDecoder", "predictive_layer"] # this is urgly
            for rmname in rm_list:  # vlist ----> vlist ----> vlist
                vlist = [tmp for tmp in vlist if rmname not in tmp.name]
            # save_params = []
            # for svname in save_list:
            #     save_params += [tmp for tmp in vlist if svname in tmp.name]
            # vlist = list(set(save_params))
            if vlist_print:
                # pdb.set_trace()
                print(rm_list,"by removing")
                # my_print("** my-note: we only update this parameters with train_strategy={}".format(train_strategy))
                my_print("\n".join([tmp.name for tmp in vlist]))
            return vlist
            
            
        vlist = get_vlist(vlist=self.model.trainable_variables, rm_list=rm_list, vlist_print=True)
        # aa = [tmp for tmp in model.trainable_variables if "bert_model" not in tmp.name]
        def _train_step(model, optimizer, example):
            example = self.preprocess_fn(example)
            
            # if self.mirrored_strategy:     
            #     vlist=var_removes(vlist=model.trainable_variables, rm_list=rm_list) 
            # else:   # self.mirrored_strategy # DEBUG: I guess we need it; but this may be slow
            #     global vlist    # NameError: name 'vlist' is not defined
            # vlist=var_removes(vlist=model.trainable_variables, rm_list=name_list) 
            vlist=get_vlist(model.trainable_variables, rm_list)

            with tf.GradientTape() as tape:
                output_dict = model(example)
                # XXX : need to exactly compute loss per gpu
                batch_size = output_dict['num_valid_turns'] if 'num_valid_turns' \
                    in output_dict else self._batch_size
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = tf.reduce_sum(value) * 1. / (batch_size * self._num_gpus)
                # pdb.set_trace()
                # tf.print(tf.shape(vlist[0]),vlist[0].name)
                grads = tape.gradient(output_dict['loss'], vlist)
                check_none_gradients(grads, vlist, model.hparams.ignore_none_gradients)
                optimizer.apply_gradients(zip(grads, vlist))
            return output_dict

        def _dist_train_step(model, optimizer, example):
            with self.mirrored_strategy.scope():
                output_dict = self.mirrored_strategy.experimental_run_v2(_train_step, (model, optimizer, example))
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = self.mirrored_strategy.reduce(
                            tf.distribute.ReduceOp.SUM, value, axis=None)
                    else:
                        value = self.mirrored_strategy.experimental_local_results(value)
                        if key == 'num_valid_turns':  # XXX : change this condition to check whether it is scalar
                            value = tf.stack(value, axis=0)
                        else:
                            value = tf.concat(value, axis=0)
                        output_dict[key] = value
            return output_dict

        # return _dist_train_step if self.mirrored_strategy else _train_step
        
        tmp_train_step = _dist_train_step if self.mirrored_strategy else _train_step
        if self.enable_function:
            self.train_step_fn = tf.function(tmp_train_step)  
        else:
            self.train_step_fn = tmp_train_step
        my_print("** we update train_step_fn with enable_function:{} and mirrored_strategy:{}".format(self.enable_function, self.mirrored_strategy))


    def _get_test_step_fn(self):
        def _test_step(model, example):
            example = self.preprocess_fn(example)
            output_dict = model(example, training=False)
            # XXX : need to exactly compute loss per gpu
            batch_size = output_dict['num_valid_turns'] if 'num_valid_turns' \
                in output_dict else self._batch_size
            for key, value in output_dict.items():
                if 'loss' in key:
                    output_dict[key] = tf.reduce_sum(value) * 1. / (batch_size * self._num_gpus)
            return output_dict

        def _dist_test_step(model, example):
            with self.mirrored_strategy.scope():
                output_dict = self.mirrored_strategy.experimental_run_v2(_test_step, (model, example))
                for key, value in output_dict.items():
                    if 'loss' in key:
                        output_dict[key] = self.mirrored_strategy.reduce(
                            tf.distribute.ReduceOp.SUM, value, axis=None)
                    else:
                        value = self.mirrored_strategy.experimental_local_results(value)
                        if key == 'num_valid_turns':
                            value = tf.stack(value, axis=0)
                        else:
                            value = tf.concat(value, axis=0)
                        output_dict[key] = value
            return output_dict

        return _dist_test_step if self.mirrored_strategy else _test_step


def run_wow_evaluation(results_dict, checkpoint_dir, mode, show_num=5):
    global_step = int(tf.compat.v1.train.get_global_step())
    if 'episode_mask' in results_dict:
        episode_mask = results_dict['episode_mask']
    else:
        episode_mask = None

    trim_fn = _trim_after_eos
    knowledge_separator = BERT_KNOWLEDGE_SEPARATOR

    predictions = trim_fn(results_dict['predictions'], mask=episode_mask)
    answers = trim_fn(results_dict['answers'], mask=episode_mask)
    contexts = trim_fn(results_dict['context'], mask=episode_mask)
    knowledge_sent_gts = trim_fn(results_dict['knowledge_sent_gt'], mask=episode_mask)
    knowledge_sent_preds = trim_fn(results_dict['knowledge_sent_pred'], mask=episode_mask)
    # knowledge_sent_gts = results_dict['knowledge_sent_gt']
    # knowledge_sent_preds = results_dict['knowledge_sent_pred']
 
    # XXX: Dump outputs
    if mode=='test':
        with open(checkpoint_dir+'_result.txt', 'w') as f:
            for know_gt, know_select, context, answer, pred in zip(knowledge_sent_gts, knowledge_sent_preds, contexts, answers, predictions):
                f.write('\n'.join([know_gt, know_select, context, answer, pred]))
                f.write('\n\n')


    # Show examples
    show_indices = random.sample(range(len(predictions)), show_num)
    # show_indices = random.sample(range(len(knowledge_sent_gts)), show_num)
    for index in show_indices:
        prediction = predictions[index]
        context = contexts[index]
        answer = answers[index]
        knowledge_sent_gt = knowledge_sent_gts[index]
        knowledge_sent_pred = knowledge_sent_preds[index]
        tqdm.write(f"{index} ({mode}).")
        tqdm.write(f"(knowledge_gt) {knowledge_sent_gt}")#knowledge
        tqdm.write(f"(knowledge_pred) {knowledge_sent_pred}")#先/后验网络
        tqdm.write(f"(ask) {context}")
        tqdm.write(f"(gt) {answer}")
        tqdm.write(f"(pred) {prediction}\n\n")

    # Evaluation
    rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer)
    perplexity = np.exp(np.mean(results_dict['gen_loss']))
    total_loss = np.mean(results_dict['loss'])
    knowledge_accuracy = accuracy_score(
        np.zeros(results_dict['knowledge_predictions'].shape, dtype=np.int32),
        results_dict['knowledge_predictions'], sample_weight=episode_mask)

    rouge_result = rouge_evaluator.run_evaluation(predictions, answers)
    loss_result = {'perplexity': perplexity,
                   'total_loss': total_loss,
                   'accuracy': knowledge_accuracy,}

    # Optional metrics
    if 'knowledge_loss' in results_dict:
        knowledge_loss = np.mean(results_dict['knowledge_loss'])
        loss_result['knowledge_loss'] = knowledge_loss
    if 'kl_loss' in results_dict:
        kl_loss = np.mean(results_dict['kl_loss'])
        loss_result['kl_loss'] = kl_loss
    if "prediction_bow_loss" in results_dict:
        prediction_bow_loss = np.mean(results_dict['prediction_bow_loss'])
        loss_result['prediction_bow_loss'] = prediction_bow_loss 
   
    if 'multi_responses' and 'multi_gt_knowledge_sentences' in results_dict:
        rouge_result, loss_result = add_multi_results(
            results_dict, rouge_result, loss_result, predictions, episode_mask, trim_fn)
            
    log_dict = {}
    log_dict.update(rouge_result)
    log_dict.update(loss_result)

    summaries = {
        f"{mode}_test_loss": loss_result,
        f"{mode}_rouge": rouge_result
    }

    return summaries, log_dict


def _trim_after_eos(sentences, replace_unk=False, mask=None):
    if mask is not None:
        # print(sentences)
        # print(mask)
        # print(sentences.shape)
        # print(mask.shape)

        assert len(sentences) == len(mask), "sentences and mask should have same length"

    trimmed_sentences = []
    
    replace_pad = False
    if sentences[0][0] == 0:
        replace_pad = True
        
    for i, sentence in enumerate(sentences):
        if mask is not None and not mask[i]:
            continue
        # Convert bytes array to utf-8 array
        sentence = np.char.decode(sentence.astype(np.bytes_), 'UTF-8')

        try:
            if replace_pad:
                eos_idx = int(np.where(sentence == data_vocab._BERT_SEP)[0][-1])
                trimmed_sentence = ' '.join(sentence[1:eos_idx])
            else:
                eos_idx = int(np.where(sentence == data_vocab._BERT_SEP)[0][0])
                trimmed_sentence = ' '.join(sentence[:eos_idx])
        except IndexError:
            trimmed_sentence = ' '.join(sentence)

        if replace_unk:
            trimmed_sentence = trimmed_sentence.replace(data_vocab._BERT_UNK, '_[UNK]')
        if replace_pad:
            trimmed_sentence = trimmed_sentence.replace(data_vocab._BERT_PAD, '')

        trimmed_sentences.append(trimmed_sentence)
    return trimmed_sentences

import json
def run_wow_evaluationv1(results_dict, checkpoint_dir, mode):
    """
    moded: choice["seen","unseen"]
    """
    global_step = int(tf.compat.v1.train.get_global_step())
    if 'episode_mask' in results_dict:
        episode_mask = results_dict['episode_mask']
    else:
        episode_mask = None

    trim_fn = _trim_after_eos
    knowledge_separator = BERT_KNOWLEDGE_SEPARATOR

    if "predictions_fst" in results_dict:   # if results_dict.get("predictions_fst", None):
        predictions_fst = trim_fn(results_dict['predictions_fst'], mask=episode_mask)
    if "pred_info" in results_dict:
        pred_info = trim_fn(results_dict['pred_info'], mask=episode_mask)         
    predictions = trim_fn(results_dict['predictions'], mask=episode_mask)
    answers = trim_fn(results_dict['answers'], mask=episode_mask)
    contexts = trim_fn(results_dict['context'], mask=episode_mask)
    # knowledge_sent_gts = trim_fn(results_dict['knowledge_sent_gt'], mask=episode_mask)
    # knowledge_sent_preds = trim_fn(results_dict['knowledge_sent_pred'], mask=episode_mask)
    knowledge_sent_gts = results_dict['knowledge_sent_gt']
    knowledge_sent_preds = results_dict['knowledge_sent_pred']

    # XXX: Dump outputs
    model_name = checkpoint_dir.split("/")[-2].strip()
    with open("my.example.{}.{}".format(model_name,mode), "w") as fp:
        for index in range(len(predictions)):
            example = {}
            prediction = predictions[index]
            context = contexts[index]
            answer = answers[index]
            knowledge_sent_gt = knowledge_sent_gts[index]
            knowledge_sent_pred = knowledge_sent_preds[index]
            example["idx"] = index
            example["ctx"] = context
            example["kld_gt"] = knowledge_sent_gt
            example["kld_pd"] = knowledge_sent_pred
            example["rsp_gt"] = answer
            example["rsp_pd"] = prediction
            if "predictions_fst" in results_dict:
                prediction_fst = predictions_fst[index]
                example["bow_pd"] = prediction_fst
            if "pred_info" in results_dict:
                pred_info_tmp = pred_info[index]
                print(pred_info_tmp)
                example["pred_info"] = pred_info_tmp            
            jstr = json.dumps(example)
            fp.write(jstr+"\n")

    # Evaluation
    rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer)
    perplexity = np.exp(np.mean(results_dict['gen_loss']))
    total_loss = np.mean(results_dict['loss'])
    knowledge_accuracy = accuracy_score(
        np.zeros(results_dict['knowledge_predictions'].shape, dtype=np.int32),
        results_dict['knowledge_predictions'], sample_weight=episode_mask)

    rouge_result = rouge_evaluator.run_evaluation(predictions, answers)
    loss_result = {'perplexity': perplexity,
                   'total_loss': total_loss,
                   'accuracy': knowledge_accuracy,}

    # Optional metrics
    if 'knowledge_loss' in results_dict:
        knowledge_loss = np.mean(results_dict['knowledge_loss'])
        loss_result['knowledge_loss'] = knowledge_loss
    if 'kl_loss' in results_dict:
        kl_loss = np.mean(results_dict['kl_loss'])
        loss_result['kl_loss'] = kl_loss
    if "prediction_bow_loss" in results_dict:
        prediction_bow_loss = np.mean(results_dict['prediction_bow_loss'])
        loss_result['prediction_bow_loss'] = prediction_bow_loss 
        prediction_vec_loss = np.mean(results_dict['prediction_vec_loss'])
        loss_result['prediction_vec_loss'] = prediction_vec_loss

    if 'multi_responses' and 'multi_gt_knowledge_sentences' in results_dict:
        rouge_result, loss_result = add_multi_results(
            results_dict, rouge_result, loss_result, predictions, episode_mask, trim_fn)

    log_dict = {}
    log_dict.update(rouge_result)
    log_dict.update(loss_result)

    summaries = {
        f"{mode}_test_loss": loss_result,
        f"{mode}_rouge": rouge_result
    }
    return summaries, log_dict

def add_multi_results(results_dict, rouge_result, loss_result, predictions, episode_mask, trim_fn):
    multi_responses = results_dict['multi_responses']
    num_responses = results_dict['num_responses'][episode_mask]
    multi_gt_knowledge_sentences = results_dict['multi_gt_knowledge_sentences']
    knowledge_sent_preds = results_dict['knowledge_sent_pred'][episode_mask]

    multi_rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1,
                                                            tokenization_fn=normalize_answer,
                                                            average=False)
    multi_rouge_results_list = []
    multi_accuracy_list = []
    for i in range(multi_responses.shape[1]):
        # choose best rouge scores among multi responses
        responses = trim_fn(multi_responses[:, i], mask=episode_mask)
        multi_rouge_result = multi_rouge_evaluator.run_evaluation(predictions, responses)
        multi_rouge_result['rouge1'][0] = multi_rouge_result['rouge1'][0] * (num_responses > i)
        multi_rouge_result['rouge2'][0] = multi_rouge_result['rouge2'][0] * (num_responses > i)
        multi_rouge_result['rougeL'][0] = multi_rouge_result['rougeL'][0] * (num_responses > i)
        multi_rouge_results_list.append(multi_rouge_result)

        # knowledge accuracy
        gt_knowledge_sentences = multi_gt_knowledge_sentences[:, i][episode_mask]
        knowledge_min_length = min(gt_knowledge_sentences.shape[-1], knowledge_sent_preds.shape[-1])
        multi_accuracy_list.append(np.logical_not(np.logical_not(
            gt_knowledge_sentences[:,:knowledge_min_length] == \
            knowledge_sent_preds[:,:knowledge_min_length]).sum(axis=1)))
    multi_rouge1_results = np.stack([x['rouge1'][0] for x in multi_rouge_results_list], axis=0)
    multi_rouge2_results = np.stack([x['rouge2'][0] for x in multi_rouge_results_list], axis=0)
    multi_rougeL_results = np.stack([x['rougeL'][0] for x in multi_rouge_results_list], axis=0)
    multi_rouge1_results = np.transpose(multi_rouge1_results, [1,0])
    multi_rouge2_results = np.transpose(multi_rouge2_results, [1,0])
    multi_rougeL_results = np.transpose(multi_rougeL_results, [1,0])
    multi_rouge1_max_indices = np.argmax(multi_rouge1_results, axis=1)
    max_multi_rouge1_results = np.max(multi_rouge1_results, axis=1)

    range_indices = np.arange(len(multi_rouge1_max_indices))
    max_multi_rouge2_results = multi_rouge2_results[range_indices, multi_rouge1_max_indices]
    max_multi_rougeL_results = multi_rougeL_results[range_indices, multi_rouge1_max_indices]

    multi_rouge1 = sum(max_multi_rouge1_results) / len(max_multi_rouge1_results)
    multi_rouge2 = sum(max_multi_rouge2_results) / len(max_multi_rouge2_results)
    multi_rougeL = sum(max_multi_rougeL_results) / len(max_multi_rougeL_results)
    rouge_result['rouge1_multi_responses'] = multi_rouge1
    rouge_result['rouge2_multi_responses'] = multi_rouge2
    rouge_result['rougeL_multi_responses'] = multi_rougeL

    # accuracy
    multi_accuracies = np.transpose(np.stack(multi_accuracy_list, axis=0), [1,0])
    multi_accuracies = multi_accuracies.sum(axis=1).astype(bool)
    multi_accuracy = sum(multi_accuracies) / len(multi_accuracies)
    loss_result['accuracy_multi_responses'] = multi_accuracy

    # perplexity
    multi_perplexity = np.exp(np.mean(results_dict['multi_gen_loss']))
    loss_result['perplexity_multi_responses'] = multi_perplexity

    return rouge_result, loss_result
__all__ = (
    'run_wow_evaluation',
    "run_wow_evaluationv1",
)
