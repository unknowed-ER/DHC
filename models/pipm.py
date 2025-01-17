import os

import tensorflow as tf
#import tensorflow_addons as tfa
import colorlog

from utils.etc_utils import NEAR_INF
from utils.config_utils import add_argument
from utils.custom_argparsers import str2bool
from data import vocabulary as data_vocab
from models import BaseModel
from models.transformer import embedding_layer
from models.transformer.transformer import TransformerDecoder
from modules.from_parlai import universal_sentence_embedding
from modules.losses import (
    masked_categorical_crossentropy,
    softmax_sequence_reconstruction_error,
    softmax_kl_divergence,
    SequenceLossLS
)
from modules.rnn import single_rnn_cell
from modules.weight_norm import WeightNormDense
from modules.discretize import gumbel_softmax
from official.bert import modeling
from official.bert import embedding_layer as bert_embedding_layer
from utils.my_print import my_print
from data.holle import _MAX_NUM_MULTI

def load_pretrained_bert_model(bert_dir, max_length):
    bert_model = modeling.get_bert_model(
        tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='input_wod_ids'),
        tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='input_mask'),
        tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='input_type_ids'),
        config=modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json')),
        float_type=tf.float32)

    # load pretrained model
    init_checkpoint = os.path.join(bert_dir, 'bert_model.ckpt')
    checkpoint = tf.train.Checkpoint(model=bert_model)
    checkpoint.restore(init_checkpoint)

    return bert_model

@add_argument("--use_copy_decoder", type=str2bool, default=True)
@add_argument("--beam_search_alpha", type=float, default=0.8)
@add_argument("--beam_size", type=int, default=1)
@add_argument("--knowledge_loss", type=float, default=0.5)
@add_argument("--num_layers", type=int, default=5)
@add_argument("--num_heads", type=int, default=2)
@add_argument("--filter_size", type=int, default=512)
@add_argument("--attention_dropout", type=float, default=0.0)
@add_argument("--relu_dropout", type=float, default=0.0)
@add_argument("--layer_postprocess_dropout", type=float, default=0.0)
@add_argument("--gumbel_temperature", type=float, default=0.5)
@add_argument("--kl_loss", type=float, default=1.0)
@add_argument("--sks_type", type=str, default="mysks")    # mysks   iclr
@add_argument("--use_pipm", type=str2bool, default=True)    
@add_argument("--prediction_update", type=str2bool, default=True)   
@add_argument("--prediction_stop_common", type=str2bool, default=True)   # now always false
@add_argument("--prediction_bow_loss_type", type=str, default="xy")    # y, xy, xyk; and other_wise.
@add_argument("--prediction_loss_alpha", type=float, default=1.0)
class PIPM(BaseModel):
    def __init__(self,
                 hparams,
                 vocabulary):
        super().__init__(hparams, vocabulary, "PIPM")
        self.masking = tf.keras.layers.Masking(mask_value=0.)

        self.encoder = load_pretrained_bert_model(self.hparams.bert_dir, self.hparams.max_length)
        self._embedding = bert_embedding_layer.EmbeddingSharedWeights(
            hparams.vocab_size, hparams.word_embed_size, self.encoder.weights[0])
        self._output_embedding = embedding_layer.EmbeddingSharedWeights(
            hparams.vocab_size, hparams.word_embed_size)
        self.decoder = TransformerDecoder(
            hparams, vocabulary, self._embedding, self._output_embedding)

        self.dialog_rnn = single_rnn_cell(hparams.word_embed_size, 'cudnn_gru', name='dialog_rnn')
        self.knowledge_flow = single_rnn_cell(hparams.word_embed_size, 'cudnn_gru', name='history_rnn')
        # self.history_query_layer = WeightNormDense(hparams.word_embed_size, use_bias=False, name='history_query_layer')
        self.prior_query_layer = WeightNormDense(hparams.word_embed_size, use_bias=False, name='prior_query_layer')
        self.posterior_query_layer = WeightNormDense(hparams.word_embed_size, use_bias=False, name='posterior_query_layer')
        if hparams.use_pipm:
            # self.prediction_layer = PredictiveLayer(hparams, embedding=self._output_embedding, my_scope="prediction")            
            self.prediction_layer = PredictiveLayer(hparams, embedding=self._embedding, my_scope="prediction")
        self.seq_loss = SequenceLossLS(
            average_across_timesteps=True,
            average_across_batch=False,
            sum_over_timesteps=False,
            sum_over_batch=False,
            label_smoothing=self.hparams.response_label_smoothing
        )
        self.test_seq_loss = SequenceLossLS(
            average_across_timesteps=True,
            average_across_batch=False,
            sum_over_timesteps=False,
            sum_over_batch=False,
            label_smoothing=0
        )

    def call(self, inputs, training: bool = True):
        context = inputs['context']
        response = inputs['response']
        knowledge_sentences = inputs['knowledge_sentences']

        episode_length = inputs['episode_length']
        context_length = inputs['context_length']
        response_length = inputs['response_length']
        knowledge_length = inputs['knowledge_sentences_length']
        num_knowledges = inputs['num_knowledge_sentences']

        max_episode_length = tf.reduce_max(episode_length)
        max_context_length = tf.reduce_max(context_length)
        max_response_length = tf.reduce_max(response_length)
        max_knowledge_length = tf.reduce_max(knowledge_length)
        max_num_knowledges = tf.reduce_max(num_knowledges)
        batch_size = self.hparams.batch_size
        episode_batch_size = batch_size * max_episode_length

        # Collapse episode_length dimension to batch
        context = tf.reshape(context, [-1, max_context_length])
        response = tf.reshape(response, [-1, max_response_length])
        knowledge_sentences = tf.reshape(knowledge_sentences, [-1, max_num_knowledges, max_knowledge_length])
        context_length = tf.reshape(context_length, [-1])
        response_length = tf.reshape(response_length, [-1])
        knowledge_length = tf.reshape(knowledge_length, [-1, max_num_knowledges])
        num_knowledges = tf.reshape(num_knowledges, [-1])

        #################
        # Encoding
        #################
        # Dialog encode (for posterior)
        # context_embedding = self._embedding(context)
        response_embedding = self._embedding(response)
        # knowledge_sentences_embedding = self._embedding(knowledge_sentences)

        _, context_outputs = self.encode(context, context_length, training)
        context_output = universal_sentence_embedding(context_outputs, context_length)

        response_mask = tf.sequence_mask(response_length, dtype=tf.float32)
        _, response_outputs = self.encode(response, response_length, training)
        response_output = universal_sentence_embedding(response_outputs, response_length)
        
        # Knowledge encode
        pooled_knowledge_embeddings, knowledge_embeddings = self.encode_knowledges(knowledge_sentences, num_knowledges, knowledge_length, training)
        knowledge_mask = tf.sequence_mask(num_knowledges, dtype=tf.bool)
        
        # Dialog encode (for posterior)
        context_response_output = tf.concat([context_output, response_output], axis=1)
        context_response_output = tf.reshape(context_response_output, [batch_size, max_episode_length, 2 * self.hparams.word_embed_size])
        dialog_outputs, dialog_state = self.dialog_rnn(context_response_output)
        dialog_outputs = tf.reshape(dialog_outputs, [episode_batch_size, self.hparams.word_embed_size])

        # Dialog encode (for prior)
        start_pad = tf.zeros([batch_size, 1, self.hparams.word_embed_size], dtype=tf.float32)
        shifted_dialog_outputs = tf.reshape(dialog_outputs, [batch_size, max_episode_length, self.hparams.word_embed_size])
        shifted_dialog_outputs = tf.concat([start_pad, shifted_dialog_outputs[:, :-1]], axis=1)
        shifted_dialog_outputs = tf.reshape(shifted_dialog_outputs, [episode_batch_size, self.hparams.word_embed_size])
        prior_dialog_outputs = tf.concat([context_output, shifted_dialog_outputs], axis=1)

        #################
        # Knowledge selection (prior & posterior)
        #################        
        if self.hparams.use_pipm:
            # Qprior, Xvec, Kvec, Knum, training=True):
            prior_dialog_outputs, pred_logits, pred_vecs = self.prediction_layer(prior_dialog_outputs, context_output, pooled_knowledge_embeddings, num_knowledges, training)
            prediction_bow_loss = get_predictive_bow_loss(self.hparams, pred_logits, context, response, knowledge_sentences[:,0])
            
        if self.hparams.sks_type == "iclr":
            my_print("** use ori sks")
            knowledge_states, prior, posterior \
                = self.sequential_knowledge_selection(pooled_knowledge_embeddings, knowledge_mask, prior_dialog_outputs, dialog_outputs, episode_length, training=training)
        elif self.hparams.sks_type == "mysks":
            my_print("** use my sks")
            knowledge_states, prior, posterior = \
                self.sks(pooled_knowledge_embeddings, knowledge_mask, prior_dialog_outputs, dialog_outputs, episode_length, training=training)
        else:
            raise NotImplementedError("we have not implement sks with type:{}".format(self.hparams.sks_type))

        prior_attentions, prior_argmaxes = prior
        posterior_attentions, posterior_argmaxes = posterior
        #
        # print("** my-note: we use posterior at testing")
        # prior_attentions, prior_argmaxes = posterior
        #################
        # Decoding
        #################
        batch_idx = tf.range(episode_batch_size, dtype=tf.int32)
        if training and self.hparams.kl_loss > 0:
            chosen_sentences_ids_with_batch = tf.stack([batch_idx, posterior_argmaxes], axis=1)
        else:
            chosen_sentences_ids_with_batch = tf.stack([batch_idx, prior_argmaxes], axis=1)
        chosen_embeddings = tf.gather_nd(knowledge_embeddings, chosen_sentences_ids_with_batch)
        chosen_sentences = tf.gather_nd(knowledge_sentences, chosen_sentences_ids_with_batch)
        knowledge_context_encoded = tf.concat([chosen_embeddings, context_outputs], axis=1)  # [batch, length, embed_dim]
        knowledge_context_sentences = tf.concat([chosen_sentences, context], axis=1)  # For masking [batch, lenth]

        #################
        # Loss
        #################
        if training:
            logits, sample_ids = self.decoder(knowledge_context_sentences, knowledge_context_encoded, response, response_embedding, training=True)
            if self.hparams.use_copy_decoder:
                response_label_smoothing = self.hparams.response_label_smoothing 
                gen_loss = softmax_sequence_reconstruction_error(
                    logits, response[:, 1:], response_length - 1, average=True,
                    average_batch=False, smoothing_rate=response_label_smoothing,
                    vocab_size=self.hparams.vocab_size)
            else:
                gen_loss = self.seq_loss(response[:, 1:], logits, response_mask[:, 1:])
            gen_loss = tf.reduce_sum(tf.reshape(gen_loss, [batch_size, max_episode_length]), axis=1) \
                / tf.cast(episode_length, tf.float32)
        else:
            logits, sample_ids = self.decoder(knowledge_context_sentences, knowledge_context_encoded, response, response_embedding, training=True)
            if self.hparams.use_copy_decoder:
                response_label_smoothing = 0.0
                gen_loss = softmax_sequence_reconstruction_error(
                    logits, response[:, 1:], response_length - 1, average=True,
                    average_batch=False, smoothing_rate=response_label_smoothing,
                    vocab_size=self.hparams.vocab_size)
            else:
                gen_loss = self.test_seq_loss(response[:, 1:], logits, response_mask[:, 1:])
            gen_loss = tf.reduce_sum(tf.reshape(gen_loss, [batch_size, max_episode_length]), axis=1) \
            / tf.cast(episode_length, tf.float32)

        # KL loss
        kl_loss = softmax_kl_divergence(prior_attentions, posterior_attentions, knowledge_mask)
        kl_loss = tf.reduce_sum(tf.reshape(kl_loss, [batch_size, max_episode_length]), axis=1) / tf.cast(episode_length, tf.float32)

        # Knowledge_loss
        answer_onehot = tf.one_hot(tf.zeros(episode_batch_size, tf.int32), max_num_knowledges)
        knowledge_loss = masked_categorical_crossentropy(
            answer_onehot, posterior_attentions, knowledge_mask,
            label_smoothing=self.hparams.knowledge_label_smoothing if training else 0)
        knowledge_loss = tf.reduce_sum(tf.reshape(knowledge_loss, [batch_size, max_episode_length]), axis=1) / tf.cast(episode_length, tf.float32)

        total_loss = gen_loss + self.hparams.kl_loss * kl_loss + self.hparams.knowledge_loss * knowledge_loss        
        if self.hparams.use_pipm:
            prediction_bow_loss = tf.reduce_sum(tf.reshape(prediction_bow_loss, [batch_size, max_episode_length]), axis=1) / tf.cast(episode_length, tf.float32)
            total_loss += prediction_bow_loss*self.hparams.prediction_loss_alpha
        else:
            prediction_bow_loss = tf.constant(0, shape=[batch_size,], dtype=tf.float32)
        # return prediction_words...
        if training:
            return {'loss': total_loss,
                    'gen_loss': gen_loss,
                    "prediction_bow_loss":prediction_bow_loss,
                    'kl_loss': kl_loss,
                    'knowledge_loss': knowledge_loss,
                    "sample_ids": tf.reshape(sample_ids, [-1])}
        else:
            test_sample_ids, test_scores, max_test_output_length = self.decoder(
                knowledge_context_sentences, knowledge_context_encoded, response, response_embedding, training=False)
            max_response_length = tf.reduce_max(response_length) - 1
            pred_words = self.pad_word_outputs(test_sample_ids, max_test_output_length)
            answer_words = self.pad_word_outputs(response[:, 1:], max_response_length)
            context_words = self.pad_word_outputs(context[:, 1:], max_context_length)

            # gt_knowledges = self.vocabulary.index_to_string(knowledge_sentences[:, 0])
            gt_knowledges = self.vocabulary.index_to_string(knowledge_sentences)
            batch_idx = tf.range(self.hparams.batch_size, dtype=tf.int32)
            pred_knowledges = self.vocabulary.index_to_string(chosen_sentences)

            episode_mask = tf.sequence_mask(episode_length)
            flat_episode_mask = tf.reshape(episode_mask, [-1])

            results_dict =  {'loss': total_loss,
                             'gen_loss': gen_loss,
                            "prediction_bow_loss":prediction_bow_loss,
                             'kl_loss': kl_loss,
                             'knowledge_loss': knowledge_loss,
                             'episode_mask': flat_episode_mask,
                             'knowledge_predictions': prior_argmaxes,
                             'knowledge_sent_gt': gt_knowledges,
                             'knowledge_sent_pred': pred_knowledges,
                             'predictions': pred_words,
                             'answers': answer_words,
                             'context': context_words}
            if self.hparams.data_name == 'holle':
                results_dict = self.add_multi_results(
                    inputs, results_dict, batch_size, episode_length,
                    max_episode_length, knowledge_context_sentences, knowledge_context_encoded
                )
            return results_dict

    def encode(self, sequence, sequence_length, training):
        # suppose that there is only 1 type embedding
        # shape of sequence: [batch_size, sequence_length, word_embed_size]
        sequence_mask = tf.sequence_mask(sequence_length, dtype=tf.int32)
        sequence_type_ids = tf.zeros(tf.shape(sequence), dtype=tf.int32)
        pooled_output, sequence_outputs = self.encoder([sequence, sequence_mask, sequence_type_ids], training)
        return pooled_output, sequence_outputs

    def encode_knowledges(self, knowledge_sentences, num_knowledges, sentences_length, training):
        max_num_knowledges = tf.reduce_max(num_knowledges)
        max_sentences_length = tf.cast(tf.reduce_max(sentences_length), dtype=tf.int32)
        episode_batch_size = tf.shape(knowledge_sentences)[0]

        squeezed_knowledge = tf.reshape(
            knowledge_sentences, [episode_batch_size * max_num_knowledges, max_sentences_length])
        squeezed_knowledge_length = tf.reshape(sentences_length, [-1])
        # squeezed_knowledge_sentences_embedding = tf.reshape(
        #     knowledge_sentences_embedding, [episode_batch_size * max_num_knowledges, max_sentences_length, self.hparams.word_embed_size])
        _, encoded_knowledge = self.encode(squeezed_knowledge, squeezed_knowledge_length, training)

        # Reduce along sequence length
        flattened_sentences_length = tf.reshape(sentences_length, [-1])
        sentences_mask = tf.sequence_mask(flattened_sentences_length, dtype=tf.float32)
        encoded_knowledge = encoded_knowledge * tf.expand_dims(sentences_mask, axis=-1)

        reduced_knowledge = universal_sentence_embedding(encoded_knowledge, flattened_sentences_length)
        embed_dim = encoded_knowledge.shape.as_list()[-1]
        reduced_knowledge = tf.reshape(reduced_knowledge, [episode_batch_size, max_num_knowledges, embed_dim])
        encoded_knowledge = tf.reshape(encoded_knowledge, [episode_batch_size, max_num_knowledges, max_sentences_length, embed_dim])

        return reduced_knowledge, encoded_knowledge

    def compute_knowledge_attention(self, knowledge, query, knowledge_mask, use_gumbel=False, training=True):
        knowledge_innerp = tf.squeeze(knowledge @ tf.expand_dims(query, axis=-1), axis=-1)
        knowledge_innerp -= tf.cast(tf.logical_not(knowledge_mask), dtype=tf.float32) * NEAR_INF  # prevent softmax from attending masked location
        knowledge_attention = tf.nn.softmax(knowledge_innerp, axis=1)
        if self.hparams.gumbel_temperature > 0 and use_gumbel:
            _, knowledge_argmax = gumbel_softmax(
                self.hparams.gumbel_temperature, probs=knowledge_attention, hard=True)
        else:
            knowledge_argmax = tf.argmax(knowledge_attention, axis=1)
        knowledge_argmax = tf.cast(knowledge_argmax, tf.int32)

        return knowledge_attention, knowledge_argmax

    def sequential_knowledge_selection(self, knowledge, knowledge_mask,
                                       prior_context, posterior_context,
                                       episode_length, training=True):
        batch_size = tf.shape(episode_length)[0]
        max_episode_length = tf.reduce_max(episode_length)
        max_num_knowledges = tf.shape(knowledge)[1]
        embed_dim = tf.shape(knowledge)[2]
        prior_embed_dim = tf.shape(prior_context)[1]
        knowledge = tf.reshape(knowledge, [batch_size, max_episode_length, max_num_knowledges, embed_dim])
        knowledge_mask = tf.reshape(knowledge_mask, [batch_size, max_episode_length, max_num_knowledges])
        prior_context = tf.reshape(prior_context, [batch_size, max_episode_length, prior_embed_dim])
        posterior_context = tf.reshape(posterior_context, [batch_size, max_episode_length, embed_dim])

        states_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        prior_attentions_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        prior_argmaxes_ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        posterior_attentions_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        posterior_argmaxes_ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        knowledge_state = tf.zeros([batch_size, embed_dim], dtype=tf.float32)

        def _loop_cond(current_episode, knowledge_state, tensorarrays, training):
            return tf.less(current_episode, max_episode_length)

        def _loop_body(current_episode, knowledge_state, tensorarrays, training):
        #for current_episode in tf.range(max_episode_length):  # For eager version
            current_knowledge_candidates = knowledge[:, current_episode]
            current_knowledge_mask = knowledge_mask[:, current_episode]

            current_prior_context = prior_context[:, current_episode]
            current_posterior_context = posterior_context[:, current_episode]

            prior_dim = self.hparams.word_embed_size * 3 if self.hparams.prediction_update else self.hparams.word_embed_size * 2
            current_prior_context.set_shape([self.hparams.batch_size, prior_dim])
            current_posterior_context.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])
            knowledge_state.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])
            # Make query
            current_prior_query = self.prior_query_layer(tf.concat([current_prior_context, knowledge_state], axis=1))
            current_posterior_query = self.posterior_query_layer(tf.concat([current_posterior_context, knowledge_state], axis=1))

            # Compute attention
            prior_knowledge_attention, prior_knowledge_argmax = self.compute_knowledge_attention(
                current_knowledge_candidates, current_prior_query, current_knowledge_mask,
                use_gumbel=False, training=training)
            batch_idx = tf.range(batch_size, dtype=tf.int32)
            posterior_knowledge_attention, posterior_knowledge_argmax = self.compute_knowledge_attention(
                current_knowledge_candidates, current_posterior_query, current_knowledge_mask,
                use_gumbel=True, training=training)

            # Sample knowledge from posterior
            chosen_sentences_id_with_batch = tf.stack([batch_idx, tf.cast(posterior_knowledge_argmax, tf.int32)], axis=1)
            chosen_knowledges = tf.gather_nd(current_knowledge_candidates, chosen_sentences_id_with_batch)
            chosen_knowledges.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])

            # Roll-out one step
            _, knowledge_state = self.knowledge_flow(
                tf.expand_dims(chosen_knowledges, axis=1),
                knowledge_state
            )
            knowledge_state.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])

            # Update TensorArray
            (states_ta, prior_attentions_ta, prior_argmaxes_ta, posterior_attentions_ta,
             posterior_argmaxes_ta) = tensorarrays
            states_ta = states_ta.write(current_episode, knowledge_state)
            prior_attentions_ta = prior_attentions_ta.write(current_episode, prior_knowledge_attention)
            prior_argmaxes_ta = prior_argmaxes_ta.write(current_episode, prior_knowledge_argmax)
            posterior_attentions_ta = posterior_attentions_ta.write(current_episode, posterior_knowledge_attention)
            posterior_argmaxes_ta = posterior_argmaxes_ta.write(current_episode, posterior_knowledge_argmax)
            tensorarrays = (states_ta, prior_attentions_ta, prior_argmaxes_ta, posterior_attentions_ta, posterior_argmaxes_ta)

            current_episode += 1

            return current_episode, knowledge_state, tensorarrays, training

        tensorarrays = (states_ta, prior_attentions_ta, prior_argmaxes_ta,
                        posterior_attentions_ta, posterior_argmaxes_ta)
        loop_vars = [tf.constant(0, dtype=tf.int32), knowledge_state, tensorarrays, training]
        loop_outputs = tf.while_loop(_loop_cond, _loop_body, loop_vars)
        (states_ta, prior_attentions_ta, prior_argmaxes_ta, posterior_attentions_ta,
         posterior_argmaxes_ta) = loop_outputs[2]

        knowledge_states = tf.reshape(tf.transpose(states_ta.stack(), perm=[1, 0, 2]), [-1, embed_dim])
        prior_attentions = tf.reshape(tf.transpose(prior_attentions_ta.stack(), perm=[1, 0, 2]), [-1, max_num_knowledges])
        prior_argmaxes = tf.reshape(tf.transpose(prior_argmaxes_ta.stack(), perm=[1, 0]), [-1])
        posterior_attentions = tf.reshape(tf.transpose(posterior_attentions_ta.stack(), perm=[1, 0, 2]), [-1, max_num_knowledges])
        posterior_argmaxes = tf.reshape(tf.transpose(posterior_argmaxes_ta.stack(), perm=[1, 0]), [-1])

        states_ta.close()
        prior_attentions_ta.close()
        prior_argmaxes_ta.close()
        posterior_attentions_ta.close()
        posterior_argmaxes_ta.close()

        return knowledge_states, (prior_attentions, prior_argmaxes), \
            (posterior_attentions, posterior_argmaxes)

    def post_sks(self, knowledge, knowledge_mask, posterior_context, episode_length, training=True):
        """(epbsz,maxnumK,768hdd) & (epbsz,maxnumK) & (epbsz, 768hdd) & (bsz,)"""
        batch_size = tf.shape(episode_length)[0]
        max_episode_length = tf.reduce_max(episode_length)
        max_num_knowledges = tf.shape(knowledge)[1]
        embed_dim = tf.shape(knowledge)[2]
        knowledge = tf.reshape(knowledge, [batch_size, max_episode_length, max_num_knowledges, embed_dim])
        knowledge_mask = tf.reshape(knowledge_mask, [batch_size, max_episode_length, max_num_knowledges])
        posterior_context = tf.reshape(posterior_context, [batch_size, max_episode_length, embed_dim])

        states_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        posterior_attentions_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        posterior_argmaxes_ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        knowledge_state = tf.zeros([batch_size, embed_dim], dtype=tf.float32)

        def _loop_cond(current_episode, knowledge_state, tensorarrays, training):
            return tf.less(current_episode, max_episode_length)

        def _loop_body(current_episode, knowledge_state, tensorarrays, training):
        #for current_episode in tf.range(max_episode_length):  # For eager version
            current_knowledge_candidates = knowledge[:, current_episode]    # [batch_size, max_num_knowledges, embed_dim]
            current_knowledge_mask = knowledge_mask[:, current_episode]     # [batch_size, max_num_knowledges]

            current_posterior_context = posterior_context[:, current_episode]   #[batch_size, embed_dim]
            # current_posterior_context.set_shape([self.hparams.batch_size, self.special_args["posterior_query_dim"]])               
            current_posterior_context.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])   
            knowledge_state.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])
            # Make query        # output: [batch_size, embed_dim]       # my-note; use knowledge state
            current_posterior_query = self.posterior_query_layer(tf.concat([current_posterior_context, knowledge_state], axis=1))

            # Compute attention
            batch_idx = tf.range(batch_size, dtype=tf.int32)
            posterior_knowledge_attention, posterior_knowledge_argmax = self.compute_knowledge_attention(
                current_knowledge_candidates, current_posterior_query, current_knowledge_mask,
                use_gumbel=True, training=training)

            # Sample knowledge from posterior   # note: posterior at this step is used for knowledge_state of next step
            chosen_sentences_id_with_batch = tf.stack([batch_idx, tf.cast(posterior_knowledge_argmax, tf.int32)], axis=1)
            chosen_knowledges = tf.gather_nd(current_knowledge_candidates, chosen_sentences_id_with_batch)
            chosen_knowledges.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])   # [batch_size, embed_dim]

            # Roll-out one step    update knowledge state #  [bsz, eps, 768hdd] & [bsz, 768hdd]
            _, knowledge_state = self.knowledge_flow(
                tf.expand_dims(chosen_knowledges, axis=1),
                knowledge_state
            )
            knowledge_state.set_shape([self.hparams.batch_size, self.hparams.word_embed_size])

            # Update TensorArray
            (states_ta, posterior_attentions_ta, posterior_argmaxes_ta) = tensorarrays
            states_ta = states_ta.write(current_episode, knowledge_state)
            posterior_attentions_ta = posterior_attentions_ta.write(current_episode, posterior_knowledge_attention)
            posterior_argmaxes_ta = posterior_argmaxes_ta.write(current_episode, posterior_knowledge_argmax)
            tensorarrays = (states_ta, posterior_attentions_ta, posterior_argmaxes_ta)

            current_episode += 1

            return current_episode, knowledge_state, tensorarrays, training

        tensorarrays = (states_ta, posterior_attentions_ta, posterior_argmaxes_ta)
        loop_vars = [tf.constant(0, dtype=tf.int32), knowledge_state, tensorarrays, training]
        loop_outputs = tf.while_loop(_loop_cond, _loop_body, loop_vars)
        (states_ta, posterior_attentions_ta, posterior_argmaxes_ta) = loop_outputs[2]
        # tensorArray is idx by episode_num, so need transpose  [maxep#, bsz, *]
        knowledge_states = tf.reshape(tf.transpose(states_ta.stack(), perm=[1, 0, 2]), [-1, embed_dim])
        posterior_attentions = tf.reshape(tf.transpose(posterior_attentions_ta.stack(), perm=[1, 0, 2]), [-1, max_num_knowledges])
        posterior_argmaxes = tf.reshape(tf.transpose(posterior_argmaxes_ta.stack(), perm=[1, 0]), [-1])

        states_ta.close()
        posterior_attentions_ta.close()
        posterior_argmaxes_ta.close()
        # [epbsz, hdd] & [epbsz, maxnumK]$ [epbsz,]
        knowledge_states.set_shape([None, self.hparams.word_embed_size])
        return knowledge_states, (posterior_attentions, posterior_argmaxes)

    def sks(self, knowledge, knowledge_mask, prior_context, posterior_context, episode_length, training=True):
        """(epbsz,maxnumK,768hdd) & (epbsz,maxnumK) & (epbsz, 768hdd*2) & (epbsz, 768hdd) & (bsz,)"""
        knowledge_states, (posterior_attentions, posterior_argmaxes) = \
                    self.post_sks(knowledge, knowledge_mask, posterior_context, episode_length, training=training)
        # need to shift knowledge_states
        # knowledge_states.set_shape([self.hparams.batch_size, None, self.hparams.word_embed_size])
        shift_knowledge_states = tf.reshape(knowledge_states, [self.hparams.batch_size, -1, self.hparams.word_embed_size])
        start_pad = tf.zeros([self.hparams.batch_size, 1, self.hparams.word_embed_size], dtype=tf.float32)
        shift_knowledge_states = tf.concat((start_pad, shift_knowledge_states[:,:-1]),axis=1)
        shift_knowledge_states = tf.reshape(shift_knowledge_states, [-1, self.hparams.word_embed_size])

        prior_query = self.prior_query_layer(tf.concat([prior_context, shift_knowledge_states], axis=1))   # [epbsz, hddhdd]
        # Compute attention
        prior_attentions, prior_argmaxes = \
            self.compute_knowledge_attention(knowledge, prior_query, knowledge_mask, use_gumbel=False, training=training)
        return knowledge_states, (prior_attentions, prior_argmaxes), (posterior_attentions, posterior_argmaxes)



    def add_multi_results(self, inputs, results_dict, batch_size, episode_length,
                          max_episode_length, knowledge_context_sentences, knowledge_context_encoded):
        responses = inputs['responses']
        gt_knowledge_sentences = inputs['gt_knowledge_sentences']

        responses_length = inputs['responses_length']
        gt_knowledge_length = inputs['gt_knowledge_sentences_length']
        num_responses = inputs['num_responses']
        num_gt_knowledges = inputs['num_gt_knowledge_sentences']

        max_responses_length = tf.reduce_max(responses_length)
        max_num_responses = tf.reduce_max(num_responses)
        max_gt_knowledge_length = tf.reduce_max(gt_knowledge_length)
        max_num_gt_knowledges = tf.reduce_max(num_gt_knowledges)

        responses = tf.reshape(responses, [-1, _MAX_NUM_MULTI, tf.shape(responses)[-1]])
        responses_length = tf.reshape(responses_length, [-1, _MAX_NUM_MULTI])
        gt_knowledge_sentences = tf.reshape(gt_knowledge_sentences, [-1, _MAX_NUM_MULTI, tf.shape(gt_knowledge_sentences)[-1]])
        gt_knowledge_length = tf.reshape(gt_knowledge_length, [-1, _MAX_NUM_MULTI])

        num_responses = tf.reshape(num_responses, [-1])
        num_gt_knowledges = tf.reshape(num_gt_knowledges, [-1])

        multi_gen_loss = self.get_multi_gen_loss(
            batch_size, episode_length, max_episode_length, responses, responses_length,
            max_num_responses, knowledge_context_sentences, knowledge_context_encoded
        )
        responses = responses[:, :, 1:]
        padding = tf.zeros([tf.shape(responses)[0],
                            tf.shape(responses)[1],
                            self.hparams.max_length - tf.shape(responses)[2] + 1],
                           dtype=tf.int64)
        multi_responses_words = self.vocabulary.index_to_string(tf.concat([tf.cast(responses, tf.int64), padding], axis=2))
        multi_gt_knowledge_words = self.vocabulary.index_to_string(gt_knowledge_sentences)
        results_dict['multi_responses'] = multi_responses_words
        results_dict['multi_gt_knowledge_sentences'] = multi_gt_knowledge_words
        results_dict['num_responses'] = num_responses
        results_dict['multi_gen_loss'] = multi_gen_loss
        return results_dict

    def get_multi_gen_loss(self, batch_size, episode_length, max_episode_length,
                           responses, responses_length, max_num_responses,
                           knowledge_context_sentences, knowledge_context_encoded):
        gen_loss_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        def _loop_cond(current_response, tensorarrays):
            return tf.less(current_response, max_num_responses)
        def _loop_body(current_response, tensorarrays):
            cand_responses_length = responses_length[:, current_response]
            max_cand_responses_length = tf.reduce_max(cand_responses_length)
            cand_responses = responses[:, current_response, :max_cand_responses_length]
            cand_responses_mask = tf.sequence_mask(cand_responses_length, dtype=tf.float32)
            cand_responses_embedding = self._embedding(cand_responses)
            multi_logits, multi_sample_ids = self.decoder(knowledge_context_sentences,
                                                          knowledge_context_encoded,
                                                          cand_responses,
                                                          cand_responses_embedding,
                                                          training=True)
            if self.hparams.use_copy_decoder:
                response_label_smoothing = 0.0
                gen_loss = softmax_sequence_reconstruction_error(
                    multi_logits, cand_responses[:, 1:], cand_responses_length - 1, average=True,
                    average_batch=False, smoothing_rate=response_label_smoothing,
                    vocab_size=self.hparams.vocab_size)
            else:
                gen_loss = self.test_seq_loss(cand_responses[:, 1:], multi_logits, cand_responses_mask[:, 1:])
            (gen_loss_ta) = tensorarrays
            gen_loss_ta = gen_loss_ta.write(current_response, gen_loss)
            tensorarrays = (gen_loss_ta)

            current_response += 1

            return current_response, tensorarrays

        tensorarrays = (gen_loss_ta)
        loop_vars = [tf.constant(0, dtype=tf.int32), tensorarrays]
        loop_outputs = tf.while_loop(_loop_cond, _loop_body, loop_vars)
        (gen_loss_ta) = loop_outputs[1]

        gen_losses = tf.transpose(gen_loss_ta.stack(), perm=[1,0])
        gen_losses = tf.reshape(gen_losses, [batch_size, max_episode_length, -1])
        best_gen_loss_index = tf.cast(tf.argmin(gen_losses + tf.cast(tf.equal(gen_losses, 0),
                                                                     dtype=tf.float32) * NEAR_INF, axis=-1), dtype=tf.int32)
        i1, i2 = tf.meshgrid(tf.range(batch_size),
                             tf.range(max_episode_length), indexing="ij")
        multi_gen_loss = tf.reduce_sum(tf.gather_nd(gen_losses, tf.stack([i1, i2, best_gen_loss_index],axis=-1)), axis=1) \
            / tf.cast(episode_length, tf.float32)

        return multi_gen_loss

from models.transformer.ffn_layer import FeedForwardNetwork
class MyFNN(FeedForwardNetwork):
    def __init__(self, hidden_size, filter_size, relu_dropout, my_scope="my"):
        """Initialize FeedForwardNetwork.

        Args:
        hidden_size: int, output dim of hidden layer.
        filter_size: int, filter size for the inner (first) dense layer.
        relu_dropout: float, dropout rate for training.
        """
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.my_scope = my_scope

    def build(self, input_shape):
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.filter_size,
            use_bias=True,
            activation=tf.nn.relu,
            name="{}/filter_layer".format(self.my_scope))
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=True, name="{}/output_layer".format(self.my_scope))
        super(FeedForwardNetwork, self).build(input_shape)

    def call(self, x, training):
        """Return outputs of the feedforward network.

        Args:
        x: tensor with shape [batch_size, length, hidden_size]
        training: boolean, whether in training mode or not.

        Returns:
        Output of the feedforward network.
        tensor with shape [batch_size, length, hidden_size]
        """
        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]

        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output,
                                    noise_shape=[batch_size, tf.shape(output)[1]],
                                    rate=self.relu_dropout)
        output = self.output_dense_layer(output)

        return output
class PredictiveLayer(tf.keras.layers.Layer):
    
    def __init__(self, hparams, embedding, my_scope="my"):
        super().__init__()
        self.hparams = hparams
        self.embedding = embedding
        self.my_scope=my_scope
    def build(self, input_shape):
        with tf.name_scope("{}/predictive_layer".format(self.my_scope)):
            self.pre_query_layer = WeightNormDense(self.hparams.word_embed_size, use_bias=False, name='pre_query_layer')
            self.pred_bow_net = MyFNN(self.hparams.vocab_size, self.hparams.filter_size, self.hparams.relu_dropout)
        super().build(input_shape)
    def call(self, Qprior, Xvec, Kvec, Knum, training=True):
        """
        Qprior: float tensor [epbsz, None]  # None = hdddim * [1,2,3]
        Xvec:   float tensor [epbsz, hdddim]
        Kvec:   float tensor [epbsz, knum, hdddim]
        Knum:   int tensor   [epbsz,]
        """
        # NEAR_INF

        """step 1. get the mixture of XXX&KKK at sentence level"""
        _kmask = tf.sequence_mask(Knum)     # bool tensor; [epbsz, knum]
        query = self.pre_query_layer(Qprior)     # [epbsz, hdddim]
        knowledge_innerp = tf.squeeze(Kvec @ tf.expand_dims(query, axis=-1), axis=-1)      # [bsz,maxnumK]    # https://github.com/tensorflow/tensorflow/issues/1062
        knowledge_innerp -= tf.cast(tf.logical_not(_kmask), dtype=tf.float32) * NEAR_INF  # prevent softmax from attending masked location
        knowledge_attention = tf.nn.softmax(knowledge_innerp, axis=1)   # [epbsz, maxnumk]
        chosen_knowledges = tf.squeeze(tf.expand_dims(knowledge_attention,1) @ Kvec, axis=1)   # [epbsz, hdd]
        xxkk_mix = tf.concat([Qprior, chosen_knowledges], axis=-1)
        """step 2. prediction """
        pred_logits = self.pred_bow_net(xxkk_mix, training)        # [epbsz, vsz]
        pred_vecs = tf.matmul(tf.nn.softmax(pred_logits, axis=-1), self.embedding.shared_weights)
        if self.hparams.prediction_update:
            Qprior = tf.concat([Qprior, pred_vecs], axis=-1)     # FIXME: [prior_query, ybow_vecs]
        
        return Qprior, pred_logits, pred_vecs
    


def get_predictive_bow_loss(hparams, pred_logits, Xin, Yin, Ktin):
    '''
    pred_logits: float tensor; [epbsz, vsz]
    Xin:         int tensor;   [epbsz, xlen]
    Yin:         int tensor;   [epbsz, ylen]    # input
    Ktin:        int tensor;   [epbsz, klen]    # input of true knowledge sentence
    '''
    bow = get_bow_from_seq(seq=Yin, vocab_size=hparams.vocab_size, stop_commons=hparams.prediction_stop_common)
    if "k" in hparams.prediction_bow_loss_type: # if hparams.prediction_bow_loss_type in ("yk", "xyk"):
        bow += get_bow_from_seq(seq=Ktin, vocab_size=hparams.vocab_size, stop_commons=hparams.prediction_stop_common)
    if "x" in hparams.prediction_bow_loss_type: # if hparams.prediction_bow_loss_type in ("xy", "xyk"):
        bow -= get_bow_from_seq(seq=Xin, vocab_size=hparams.vocab_size, stop_commons=hparams.prediction_stop_common)
        bow = bow * tf.cast((bow>0), dtype=tf.int32)
    bow_labels = soft_bow_to_distribution(bow)
    pred_bow_loss = tf.nn.softmax_cross_entropy_with_logits(bow_labels, logits=pred_logits)
    return pred_bow_loss


def get_bow_from_seq(seq, vocab_size, stop_commons=True):
    """[bsz, slen]"""   
    if stop_commons:
        # 0-pad; 100-unk; 101-cls; 102-sep; 103-mask; 1996-the, 2019-an; and punctuation(999, 1700). not use so much
        bert_stop_ids = [0,100,101,102,103, 1996, 2019]+list(range(999,1200))       # 
        stop_dims = 2500
        bert_stop_mask = tf.one_hot(bert_stop_ids, depth=stop_dims, dtype=tf.int32)     # [nnn, stop_dims]
        bert_stop_mask = tf.reduce_sum(bert_stop_mask, axis=0)
        bert_stop_mask = tf.concat([bert_stop_mask, tf.zeros([vocab_size-stop_dims], dtype=tf.int32)],axis=0)   # [vsz]
        bert_stop_mask = 1 - bert_stop_mask     # 0 for stop_words.
    else:           # pad,unk,cls,seq,mask,",","-","."
        # bert_stop_ids = [0,100,101,102,103]
        bert_stop_ids = [0,100,101,102,103,1011,1012,1013]
        bert_stop_mask = tf.reduce_sum(tf.one_hot(bert_stop_ids, depth=vocab_size, dtype=tf.int32), axis=0) # [vsz]
        bert_stop_mask = 1 - bert_stop_mask

    bow = tf.one_hot(seq, depth=vocab_size, dtype=tf.int32)  # [epbsz, len, vsz]
    bow = tf.reduce_sum(bow, axis=1)      # [epbsz, vsz]
    bow = bow * tf.expand_dims(bert_stop_mask, axis=0)
    return bow      # [bsz, vsz]

def soft_bow_to_distribution(bow):
    """[bsz, vsz]"""
    # first elem is pad_token, to avoid the zero-div
    bow = tf.concat([tf.expand_dims(tf.ones_like(bow[:,0]), axis=-1), bow[:,1:]], axis=-1)
    labels = bow / tf.expand_dims(tf.reduce_sum(bow, axis=-1), axis=-1)   # [epbsz, vsz]
    return labels

def tf_cosine_dist(aa, bb, use_normalized=True):
    aa_norm = tf.math.l2_normalize(aa, axis=-1)
    bb_norm = tf.math.l2_normalize(bb, axis=-1)
    simi = tf.reduce_sum(aa_norm*bb_norm, axis=-1)
    if use_normalized:
        dist = (1-simi)/ 2.0
    else:
        dist = 1-simi
    return dist
