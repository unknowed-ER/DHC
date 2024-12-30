import random
import tensorflow as tf

class GlobalAggregator(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout_rate):
        super(GlobalAggregator, self).__init__()
        self.dim = hidden_size
        self.dropout = dropout_rate

        # self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        # self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        # self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        # self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.w_1 = tf.Variable(tf.random.truncated_normal([2*self.dim, self.dim]))
        self.w_2 = tf.Variable(tf.random.truncated_normal([self.dim, 1]))
        self.w_3 = tf.Variable(tf.random.truncated_normal([2*self.dim, self.dim]))
        self.bias = tf.Variable(tf.random.truncated_normal([self.dim]))

    def call(self, self_vector, neighbor_vector, extra_vector=None, batch_size=None, agg_num=-1):
        if extra_vector is not None:
            # alpha = torch.matmul(torch.cat([extra_vector, neighbor_vector], -1), self.w_1).squeeze(-1)
            # alpha = F.leaky_relu(alpha, negative_slope=0.2)
            # alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            # alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            # neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)

            # alpha = F.cosine_similarity(extra_vector, neighbor_vector, dim=-1).unsqueeze(-1)
            alpha = tf.keras.losses.cosine_similarity(extra_vector, neighbor_vector, axis=-1)
            alpha = tf.expand_dims(alpha, -1)
            
            # _mask = -9e15 * torch.ones_like(alpha)
            # one_mask = torch.ones_like(alpha)
            # # alpha = torch.where(alpha > 0.3, alpha, _mask).squeeze()
            
            # alpha = torch.softmax(alpha.squeeze(-1), -1).unsqueeze(-1)
            # neighbor_vector = torch.sum(alpha * neighbor_vector, -2)
            alpha = tf.nn.softmax(tf.squeeze(alpha, axis=-1), axis=-1)
            alpha = tf.expand_dims(alpha, axis=-1)
            neighbor_vector = tf.reduce_sum(alpha * neighbor_vector, axis=-2)
        else:
            # neighbor_vector = torch.mean(neighbor_vector, dim=-2)
            neighbor_vector = tf.reduce_mean(neighbor_vector, axis=-2)
        
        # # output = self_vector + neighbor_vector
        # output = torch.cat([self_vector, neighbor_vector], -1)
        output = tf.concat([self_vector, neighbor_vector], -1)

        # # output = F.dropout(output, p=0.5, training=self.training)
        # output = torch.matmul(output, self.w_3)
        # output = output.view(self_vector.shape[0], -1, self.dim)
        # output = torch.tanh(output)
        output = tf.matmul(output, self.w_3)
        output = tf.reshape(output, [batch_size, -1, self.dim])
        # output = tf.reshape(output, [self_vector.shape[0], agg_num, self.dim])
        # output = tf.reshape(output, [self_vector.shape[0], -1, self.dim])# FIXME: 1 or 12
        output = tf.tanh(output)

        return self_vector

class GlobalSage(tf.keras.layers.Layer):
    def __init__(self, num_layers, hidden_size, adj, sample_num):
        super(GlobalSage, self).__init__()
        self.dim = hidden_size
        self.hop = num_layers
        self.hidden_size = hidden_size
        self.neighbor_num = sample_num
        self.adj = adj
        self.global_gnn = []
        for i in range(self.hop):
            agg = GlobalAggregator(self.hidden_size, 0.5)
            # self.add_module('agg_gcn_{}'.format(i), agg)
            agg._name = agg.name+str('_{}'.format(i))
            self.global_gnn.append(agg)

    def call(self, nodes, embedding):
        # sub edge
        adj = []
        for i in self.adj:
            # top sample_num
            # adj.append(i[:self.neighbor_num] + [0]*(self.neighbor_num-len(i[:self.neighbor_num])))
            # random choice
            adj.append(random.sample(i, min(self.neighbor_num, len(i))) + [0]*(self.neighbor_num-len(i[:self.neighbor_num])))
        adj = tf.constant(adj, dtype=tf.int32)

        batch_size = len(nodes)
        item_neighbors = [tf.expand_dims(nodes,1)]
        support_size = 1

        for i in range(1, self.hop + 1):
            item_sample_i = tf.gather_nd(adj, tf.expand_dims(item_neighbors[-1], -1))
            support_size *= self.neighbor_num
            item_neighbors.append(tf.reshape(item_sample_i, [batch_size, support_size]))

        # XXX: to parse 0 as neighbor
        entity_vectors = [embedding(i) for i in item_neighbors]
        # entity_vectors = tf.gather_nd(adj, )

        shape1 = [batch_size, -1, self.hidden_size]
        shape2 = [batch_size, -1, self.neighbor_num, self.hidden_size]
        # new_hidden = self.global_gnn[0](hidden, entity_vectors[1].view(shape1), weight_vectors[0].view(shape2),
        #                                 hidden.unsqueeze(2).repeat(1, 1, self.sample_num, 1), inputs.gt(0))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            for hop in range(self.hop - n_hop):
                aggregator = self.global_gnn[n_hop]
                a = entity_vectors[hop]
                b = entity_vectors[hop+1]
                x = tf.expand_dims(entity_vectors[hop], -2)
                s = tf.constant([1, 1, self.neighbor_num, 1])
                c = tf.tile(x, s)
                vector = aggregator(self_vector=tf.reshape(a, shape1),
                                    neighbor_vector=tf.reshape(b, shape2),
                                    extra_vector=tf.reshape(c, shape2),
                                    # extra_vector=None, # to mean neighbors
                                    batch_size=batch_size,
                                    agg_num=pow(12, hop)# 12/64
                                    )
                v = tf.norm(vector, ord=2, axis=2, keepdims=True) + 1e-8
                # v = vector.norm(p=2, dim=2, keepdim=True) + 1e-8
                vector = vector / v
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        new_hidden = tf.reshape(entity_vectors[0], [batch_size, self.hidden_size])
        # new_hidden = entity_vectors[0].view(batch_size, self.hidden_size)

        return new_hidden
