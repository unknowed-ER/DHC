import os

import tensorflow as tf
#import tensorflow_addons as tfa
import colorlog

from utils.etc_utils import NEAR_INF
from utils.config_utils import add_argument
from utils.custom_argparsers import str2bool
from data import vocabulary as data_vocab
from data.holle import _MAX_NUM_MULTI
from models import BaseModel
from models.transformer import embedding_layer
from models.transformer import attention_layer
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

import pandas as pd 
import numpy as np 
# from models import GCN
import tf_geometric as tfg
from models.attention import MultiHeadAttention
from models.graphsage import GlobalSage
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization

# for graph in test_data:
#     # 使用缓存cache可以避免对归一化边信息的重复计算，大幅度加速GCN的计算
#     outputs = gcn_layer([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
#     print(outputs)

# # OOP Style GAT (Multi-head Graph Attention Network)
# # 面向对象风格的多头图注意力网络GAT
# for graph in test_data:
#     outputs = gat_layer([graph.x, graph.edge_index])
#     print(outputs)

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
class SKT_KG(BaseModel):
    def __init__(self,
                 hparams,
                 vocabulary,
                 adj=None):
        super().__init__(hparams, vocabulary, "SKT_KG")
        self.masking = tf.keras.layers.Masking(mask_value=0.)

        self.encoder = load_pretrained_bert_model(self.hparams.bert_dir, self.hparams.max_length)
        self._embedding = bert_embedding_layer.EmbeddingSharedWeights(
            hparams.vocab_size, hparams.word_embed_size, self.encoder.weights[0])
        self._output_embedding = embedding_layer.EmbeddingSharedWeights(
            hparams.vocab_size, hparams.word_embed_size)
        self.decoder = TransformerDecoder(
            hparams, vocabulary, self._embedding, self._output_embedding)

        self.dialog_rnn = single_rnn_cell(
            hparams.word_embed_size, 'cudnn_gru', name='dialog_rnn')
        self.history_rnn = single_rnn_cell(
            hparams.word_embed_size, 'cudnn_gru', name='history_rnn')
        self.history_query_layer = WeightNormDense(hparams.word_embed_size, use_bias=False, name='history_query_layer')
        self.prior_query_layer = WeightNormDense(hparams.word_embed_size, use_bias=False, name='prior_query_layer')
        self.posterior_query_layer = WeightNormDense(hparams.word_embed_size, use_bias=False, name='posterior_query_layer')

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

        # 知识图谱引入

        # GCN or GAT
        # self.GRAPH_DIM = 768 # 128
        self.GRAPH_DIM = 128
        # self.gcn_layer = tfg.layers.GCN(units=self.GRAPH_DIM, activation=tf.nn.relu)## dis
        # self.gcn_layer.build([[76265,128]])
        # self.gat_layer = tfg.layers.GAT(units=128, activation=tf.nn.relu, num_heads=4)
        # self.gat_layer.build(input_shapes=128)

        # nodes
        # def _create_entity_embeddings(entity_num, embedding_size, padding_idx):
        #     """Create and initialize word embeddings."""
        #     e = nn.Embedding(entity_num, embedding_size)
        #     nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
        #     nn.init.constant_(e.weight[padding_idx], 0)
        #     return e

        # self.concept_embeddings=tf.random.truncated_normal([12996+1,self.GRAPH_DIM], mean=0, stddev=self.GRAPH_DIM**-0.5)
        # self.concept_embeddings=tf.Variable(self.concept_embeddings)

        self.concept_embeddings_layer=tf.keras.layers.Embedding(
            12996+1, 128, embeddings_initializer='normal', mask_zero=True)
        self.concept_embeddings_layer.build(())
        self.concept_embeddings = self.concept_embeddings_layer.embeddings
        
        # edges
        def concept_edge_list4GCN():
            node2index = {line.strip().split('\t')[0]:int(line.strip().split('\t')[1]) for line in open('kgtmp/entity2index.txt',encoding='utf-8')}
            f=open('kgtmp/edges.txt',encoding='utf-8')
            edges=set()
            stopwords=set([word.strip() for word in open('kgtmp/stopwords.txt',encoding='utf-8')])
            for line in f:
                lines=line.strip().split('\t')
                entity0=node2index[lines[1].split('/')[0]]
                entity1=node2index[lines[2].split('/')[0]]
                if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
                    continue
                edges.add((entity0,entity1))
                edges.add((entity1,entity0))
            edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
            return tf.constant(edge_set, dtype=tf.int32)
        self.concept_edge_sets=concept_edge_list4GCN()
        
        self.ffn = tf.keras.Sequential()
        for size in [768]:
            self.ffn.add(Dense(size, activation='relu'))
        self.kg_attention_encode = MultiHeadAttention(
            hparams.word_embed_size, hparams.num_heads,
            # hparams.attention_dropout
        )
        self.norm2 = LayerNormalization()
        # self.dropout = tf.keras.layers.Dropout(.2, input_shape=(2,))
        self.dropout = tf.keras.layers.Dropout(.2)

        # graph sage (for unsupervised graph representation learning)
        self.Q = 1
        hop, hidden_size, adjust_list, sample_num = 2, 128, adj, 64#12
        self.global_gnn = GlobalSage(hop, hidden_size, adjust_list, sample_num)

    def norm(self, inputs, epsilon = 1e-8, scope="ln"):
        '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        
        Returns:
        A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.name_scope(scope):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            beta= tf.compat.v1.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.compat.v1.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
        return outputs

    # Unsupervised loss & forward
    def get_loss_sage(self, embeddings, unique_nodes_batch, positive_node_pairs, negative_node_pairs):
        # node2index_o = {n: i for i, n in enumerate(a)}
        
        # node2index = tf.zeros([tf.reduce_max(unique_nodes_batch)+1], tf.int32)

        # idx = tf.expand_dims(tf.cast(unique_nodes_batch, dtype=tf.int64), -1)
        # value = tf.range(0, unique_nodes_batch.shape[0], dtype=tf.int64)
        # shape = [tf.reduce_max(unique_nodes_batch)+1]
        # sparse = tf.SparseTensor(indices=idx, values=value, dense_shape=shape)
        # sparse = tf.sparse.reorder(sparse)
        # node2index = tf.sparse.to_dense(sparse)

        # df  = pd.DataFrame(node2index)
        # # Create a new set of index from min and max of the dictionary keys. 
        # new_index = np.arange( int(df.index.min()),
        #                     int(df.index.max()+1)).astype(str)
        # # Add the new index to the existing index and fill the nan values with 0, take a transpose of dataframe. 
        # new_df = df.reindex(new_index).fillna(0).T.astype(int)

        # nodes_score = []
        # pos_scores = []
        # pos_scores2 = []
        # neg_scores = []

        # for pps, nps in zip(positive_node_pairs, negative_node_pairs):
        #     # Q * Exception(negative score)
        #     # indexs = [list(x) for x in zip(*nps)]
        #     node_index = [node2index[nps[0]]]
        #     neighb_index = [node2index[nps[1]]]
        #     # node_indexs = [node2index[x] for x in indexs[0]]
        #     # neighb_indexs = [node2index[x] for x in indexs[1]]
        #     # neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            
        #     # node = embeddings[node2index[indexs[0][0]]].view(1, -1)
        #     # node = tf.reshape(embeddings[node2index[indexs[0][0]]], [1, -1])
        #     # neighb = embeddings[1:]
        #     # neg_score = torch.matmul(F.normalize(node), F.normalize(neighb).t())
        #     # neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
        #     node = tf.gather_nd(embeddings, tf.expand_dims(node_index, -1))
        #     neighb = tf.gather_nd(embeddings, tf.expand_dims(neighb_index, -1))
        #     neg_score = tf.matmul(tf.math.l2_normalize(node), tf.transpose(tf.math.l2_normalize(neighb)))
        #     # neg_score = self.Q * tf.reduce_mean(tf.math.log(tf.sigmoid(-neg_score)), 0) # for batch (node, nei1, nei2, ...)
        #     neg_score = tf.math.log(tf.sigmoid(-neg_score)) # for single example
        #     neg_score = tf.reduce_mean(neg_score, 0)
        #     neg_scores.append(neg_score)
        #     # neg_score = self.Q * tf.reduce_mean(neg_score, 0, keepdims=True)

        #     # multiple positive score
        #     node_index = [node2index[pps[0]]]
        #     neighb_index = [node2index[pps[1]]]
        #     # indexs = [list(x) for x in zip(*pps)]
        #     # node_indexs = [node2index[x] for x in indexs[0]]
        #     # neighb_indexs = [node2index[x] for x in indexs[1]]
        #     # pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
        #     # pos_score = torch.log(torch.sigmoid(pos_score))
        #     node = tf.gather_nd(embeddings, tf.expand_dims(node_index, -1))
        #     neighb = tf.gather_nd(embeddings, tf.expand_dims(neighb_index, -1))
        #     pos_score = tf.keras.losses.cosine_similarity(node, neighb)
        #     pos_scores2.append(pos_score)
        #     pos_score = tf.math.log(tf.sigmoid(pos_score))
        #     pos_score = tf.reduce_mean(pos_score, 0)

        #     # nodes_score.append(torch.mean(- neg_score).view(1, -1))
        #     # nodes_score.append(tf.reduce_mean(-neg_score), [1,-1])# for batch
        #     nodes_score.append(-pos_score-neg_score)
        #     pos_scores.append(pos_score)

        # static test
        # pos = 
        # node = tf.gather_nd(node2index, tf.expand_dims(negative_node_pairs[:, 0], -1))# translate index
        # neibor = tf.gather_nd(node2index, tf.expand_dims(negative_node_pairs[:, 1], -1))
        # node = tf.gather_nd(embeddings, tf.expand_dims(node, -1))
        # neibor = tf.gather_nd(embeddings, tf.expand_dims(neibor, -1))
        node = tf.gather_nd(embeddings, tf.expand_dims(negative_node_pairs[:, 0], -1))
        neibor = tf.gather_nd(embeddings, tf.expand_dims(negative_node_pairs[:, 1], -1))
        neg_score_new_n = -1*tf.reduce_sum(tf.multiply(tf.math.l2_normalize(node,-1), tf.math.l2_normalize(neibor,-1)), -1)
        neg_score_new = tf.math.log(tf.sigmoid(neg_score_new_n))
        neg_s = tf.reduce_mean(neg_score_new)
        
        # node = tf.gather_nd(node2index, tf.expand_dims(positive_node_pairs[:, 0], -1))
        # neibor = tf.gather_nd(node2index, tf.expand_dims(positive_node_pairs[:, 1], -1))
        # node = tf.gather_nd(embeddings, tf.expand_dims(node, -1))
        # neibor = tf.gather_nd(embeddings, tf.expand_dims(neibor, -1))
        node = tf.gather_nd(embeddings, tf.expand_dims(positive_node_pairs[:, 0], -1))
        neibor = tf.gather_nd(embeddings, tf.expand_dims(positive_node_pairs[:, 1], -1))
        pos_s = tf.reduce_mean(tf.math.log(tf.sigmoid(tf.reduce_sum(tf.multiply(tf.math.l2_normalize(node, axis=-1), tf.math.l2_normalize(neibor, axis=-1)),-1))))
        pass

        # loss = torch.mean(torch.cat(nodes_score, 0), 0)
        # loss = tf.reduce_mean(tf.concat(nodes_score, 0), 0)
        loss = -(neg_s+pos_s)*5
        return loss

    def compute_global_loss(self, inputs, lam=1.0):
        positive_pairs = inputs['positive_pairs']
        negative_pairs = inputs['negative_pairs']

        cat = tf.concat([positive_pairs, negative_pairs],0)
        batch_nodes, idx = tf.unique(tf.reshape(cat, [-1]))
        idx = tf.reshape(idx, [-1, 2])
        s = positive_pairs.shape[0]
        positive_pairs = idx[:s]
        negative_pairs = idx[s:]
        # batch_nodes_v1 = list(set([i for i in tf.reshape(positive_pairs, [-1]).numpy()]) | set([i for i in tf.reshape(negative_pairs, [-1]).numpy()]))# edge to nodes, 去重
        embedding_nodes = self.global_gnn(batch_nodes, self.concept_embeddings_layer)# GraphSage
        loss_unspurvised = lam * self.get_loss_sage(embedding_nodes, batch_nodes, positive_pairs, negative_pairs)

        return {'loss': loss_unspurvised}

    def get_hidden(self, inputs, training=False):
        context = inputs['context']
        response = inputs['response']
        knowledge_sentences = inputs['knowledge_sentences']
        knowledge_kg_masks = inputs['knowledge_kg_masks']

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
        knowledge_kg_masks = tf.reshape(knowledge_kg_masks, [-1, max_num_knowledges, max_knowledge_length])
        context_length = tf.reshape(context_length, [-1])
        response_length = tf.reshape(response_length, [-1])
        knowledge_length = tf.reshape(knowledge_length, [-1, max_num_knowledges])
        num_knowledges = tf.reshape(num_knowledges, [-1])

        # KG Encoding
        # con_nodes_features=self.gcn_layer([self.concept_embeddings,self.concept_edge_sets])
        # graph_con_embs = tf.gather_nd(con_nodes_features, indices=tf.expand_dims(knowledge_kg_masks, -1))
        # GraphSage
        s = tf.shape(knowledge_kg_masks)
        node_index, idx = tf.unique(tf.reshape(knowledge_kg_masks, [-1]))
        con_nodes_features = self.global_gnn(node_index, self.concept_embeddings_layer)
        graph_con_embs = tf.reshape(tf.gather_nd(con_nodes_features, tf.expand_dims(idx, -1)), tf.concat([s,[128]],0))
        # -----------

        #################
        # Encoding
        #################
        # Dialog encode (for posterior)
        context_embedding = self._embedding(context)
        response_embedding = self._embedding(response)
        knowledge_sentences_embedding = self._embedding(knowledge_sentences)

        _, context_outputs = self.encode(context, context_length, training)
        context_output = universal_sentence_embedding(context_outputs, context_length)

        response_mask = tf.sequence_mask(response_length, dtype=tf.float32)
        _, response_outputs = self.encode(response, response_length, training)
        response_output = universal_sentence_embedding(response_outputs, response_length)

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

        # Knowledge encode
        pooled_knowledge_embeddings, knowledge_embeddings = self.encode_knowledges(
            knowledge_sentences, graph_con_embs, knowledge_kg_masks,
            num_knowledges, knowledge_length, knowledge_sentences_embedding, training)
        knowledge_mask = tf.sequence_mask(num_knowledges, dtype=tf.bool)

        # Knowledge selection (prior & posterior)
        knowledge_states, prior, posterior = self.sequential_knowledge_selection(
            pooled_knowledge_embeddings, knowledge_mask,
            prior_dialog_outputs, dialog_outputs, episode_length,
            training=training
        )

        prior_attentions, prior_argmaxes = prior
        posterior_attentions, posterior_argmaxes = posterior

        return prior, posterior
        


    def call(self, inputs, training: bool = True):
        context = inputs['context']
        response = inputs['response']
        knowledge_sentences = inputs['knowledge_sentences']
        knowledge_kg_masks = inputs['knowledge_kg_masks']

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
        knowledge_kg_masks = tf.reshape(knowledge_kg_masks, [-1, max_num_knowledges, max_knowledge_length])
        context_length = tf.reshape(context_length, [-1])
        response_length = tf.reshape(response_length, [-1])
        knowledge_length = tf.reshape(knowledge_length, [-1, max_num_knowledges])
        num_knowledges = tf.reshape(num_knowledges, [-1])

        # KG Encoding
        # con_nodes_features=self.gcn_layer([self.concept_embeddings,self.concept_edge_sets])
        # graph_con_embs = tf.gather_nd(con_nodes_features, indices=tf.expand_dims(knowledge_kg_masks, -1))
        # GraphSage
        s = tf.shape(knowledge_kg_masks)
        node_index, idx = tf.unique(tf.reshape(knowledge_kg_masks, [-1]))
        con_nodes_features = self.global_gnn(node_index, self.concept_embeddings_layer)
        graph_con_embs = tf.reshape(tf.gather_nd(con_nodes_features, tf.expand_dims(idx, -1)), tf.concat([s,[128]],0))
        # -----------

        #################
        # Encoding
        #################
        # Dialog encode (for posterior)
        context_embedding = self._embedding(context)
        response_embedding = self._embedding(response)
        knowledge_sentences_embedding = self._embedding(knowledge_sentences)

        _, context_outputs = self.encode(context, context_length, training)
        context_output = universal_sentence_embedding(context_outputs, context_length)

        response_mask = tf.sequence_mask(response_length, dtype=tf.float32)
        _, response_outputs = self.encode(response, response_length, training)
        response_output = universal_sentence_embedding(response_outputs, response_length)

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

        # Knowledge encode
        pooled_knowledge_embeddings, knowledge_embeddings = self.encode_knowledges(
            knowledge_sentences, graph_con_embs, knowledge_kg_masks,
            num_knowledges, knowledge_length, knowledge_sentences_embedding, training)
        knowledge_mask = tf.sequence_mask(num_knowledges, dtype=tf.bool)

        # Knowledge selection (prior & posterior)
        knowledge_states, prior, posterior = self.sequential_knowledge_selection(
            pooled_knowledge_embeddings, knowledge_mask,
            prior_dialog_outputs, dialog_outputs, episode_length,
            training=training
        )

        prior_attentions, prior_argmaxes = prior
        posterior_attentions, posterior_argmaxes = posterior

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
            logits, sample_ids = self.decoder(knowledge_context_sentences, knowledge_context_encoded,
                                 response, response_embedding, training=True)
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
            logits, sample_ids = self.decoder(knowledge_context_sentences, knowledge_context_encoded,
                                              response, response_embedding, training=True)
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
        kl_loss = tf.reduce_sum(tf.reshape(kl_loss, [batch_size, max_episode_length]), axis=1) \
            / tf.cast(episode_length, tf.float32)

        # Knowledge_loss
        answer_onehot = tf.one_hot(tf.zeros(episode_batch_size, tf.int32), max_num_knowledges)
        knowledge_loss = masked_categorical_crossentropy(
            answer_onehot, posterior_attentions, knowledge_mask,
            label_smoothing=self.hparams.knowledge_label_smoothing if training else 0)
        knowledge_loss = tf.reduce_sum(tf.reshape(knowledge_loss, [batch_size, max_episode_length]), axis=1) \
            / tf.cast(episode_length, tf.float32)

        total_loss = gen_loss + self.hparams.kl_loss * kl_loss + self.hparams.knowledge_loss * knowledge_loss

        if training:
            return {'loss': total_loss,
                    'gen_loss': gen_loss,
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
                             'kl_loss': kl_loss,
                             'knowledge_loss': knowledge_loss,
                             'episode_mask': flat_episode_mask,
                             'knowledge_predictions': prior_argmaxes,
                             'knowledge_sent_gt': gt_knowledges,
                             'knowledge_sent_pred': pred_knowledges,
                             'predictions': pred_words,
                             'answers': answer_words,
                             'context': context_words}

            # code for holle multi responses
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



    def encode_mutual(self, sequence, sequence_length, training, graph_con_embs, graph_con_embs_mask):
        sequence_mask = tf.sequence_mask(sequence_length, dtype=tf.int32)
        sequence_type_ids = tf.zeros(tf.shape(sequence), dtype=tf.int32)
        pooled_output, sequence_outputs = self.encoder([sequence, sequence_mask, sequence_type_ids], training)

        # NOTE: Mutual Attention with KG embeddings from GCN
        episode_batch_size = tf.shape(sequence)[0]
        squeezed_kg_emb = tf.reshape(
            graph_con_embs, [episode_batch_size, sequence_length, self.GRAPH_DIM])
        # fake 768
        squeezed_kg_emb = tf.tile(squeezed_kg_emb, [1,1,6])
        graph_con_embs_mask = tf.reshape(
            graph_con_embs_mask, [episode_batch_size, sequence_length])
        # encoded_knowledge += squeezed_kg_emb
        x = self.kg_attention_encode(
            [sequence_outputs,
            squeezed_kg_emb,
            squeezed_kg_emb,
            graph_con_embs_mask]
            # q=encoded_knowledge,
            # k=squeezed_kg_emb,
            # v=squeezed_kg_emb,
            # mask=graph_con_embs_mask
        )
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = sequence_outputs + x

        return x



    def encode_knowledges(self, knowledge_sentences, graph_con_embs, graph_con_embs_mask, num_knowledges, sentences_length,
                          knowledge_sentences_embedding, training):
        max_num_knowledges = tf.reduce_max(num_knowledges)
        max_sentences_length = tf.cast(tf.reduce_max(sentences_length), dtype=tf.int32)
        episode_batch_size = tf.shape(knowledge_sentences)[0]

        squeezed_knowledge = tf.reshape(
            knowledge_sentences, [episode_batch_size * max_num_knowledges, max_sentences_length])
        squeezed_knowledge_length = tf.reshape(sentences_length, [-1])
        squeezed_knowledge_sentences_embedding = tf.reshape(
            knowledge_sentences_embedding, [episode_batch_size * max_num_knowledges, max_sentences_length, self.hparams.word_embed_size])
        _, encoded_knowledge = self.encode(squeezed_knowledge, squeezed_knowledge_length, training)

        # NOTE: Mutual Attention with KG embeddings from GCN
        squeezed_kg_emb = tf.reshape(
            graph_con_embs, [episode_batch_size * max_num_knowledges, max_sentences_length, self.GRAPH_DIM])
        # fake 768
        squeezed_kg_emb = tf.tile(squeezed_kg_emb, [1,1,6])
        graph_con_embs_mask = tf.reshape(
            graph_con_embs_mask, [episode_batch_size * max_num_knowledges, max_sentences_length])
        # encoded_knowledge += squeezed_kg_emb
        x = self.kg_attention_encode(
            [encoded_knowledge,
            squeezed_kg_emb,
            squeezed_kg_emb,
            graph_con_embs_mask]
            # q=encoded_knowledge,
            # k=squeezed_kg_emb,
            # v=squeezed_kg_emb,
            # mask=graph_con_embs_mask
        )
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = encoded_knowledge + x
        # a = self.norm2(x)
        # b = self.norm(x)
        # c = tf.math.l2_normalize(x, -1)
        encoded_knowledge = x
        # --------------------------------------------------

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

            current_prior_context.set_shape([self.hparams.batch_size, self.hparams.word_embed_size * 2])
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
            _, knowledge_state = self.history_rnn(
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