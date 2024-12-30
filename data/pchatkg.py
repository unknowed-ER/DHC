from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import json
import pickle
import os
from collections import namedtuple
import random
import colorlog
from operator import itemgetter

import tensorflow as tf
import numpy as np
import colorlog
from tqdm import tqdm
from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import create_task

from data.dataset_reader import (
    DatasetReader, string_split, list_of_string_split, bucketing,
    list_of_list_of_string_split, tensor_pad,
    _scalar, _vector, _matrix, _tensor,
)
from data import vocabulary as data_vocab

from official.bert import tokenization

PARLAI_KNOWLEDGE_SEPARATOR = '__knowledge__'
BERT_KNOWLEDGE_SEPARATOR = '_ _ knowledge _ _'


class PchatKGDatasetReader(DatasetReader):
    iterator_shapes = {
        "context": _matrix(),
        "response": _matrix(),
        "chosen_topic": _matrix(),
        "knowledge_sentences": _tensor(),
        "knowledge_kg_masks": _tensor(),
        "episode_length": _scalar(),
    }
    iterator_types = {
        "context": tf.string,
        "response": tf.string,
        "chosen_topic": tf.string,
        "knowledge_sentences": tf.string,
        "knowledge_kg_masks": tf.string,
        "episode_length": tf.int32,
    }

    def __init__(self,
                 batch_size: int,
                 num_epochs: int,
                 buffer_size: int = 5000,
                 bucket_width: int = 5,
                 max_length: int = 51,
                 max_episode_length: int = 5,
                 max_knowledge: int = 32,
                 knowledge_truncate: int = 34,
                 cache_dir: str = None,
                 pad_to_max: bool = True,
                 bert_dir: str = None) -> None:
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._buffer_size = buffer_size
        self._bucket_width = bucket_width
        self._max_length = max_length
        self._max_episode_length = max_episode_length
        self._max_knowledge = max_knowledge
        self._knowledge_truncate = knowledge_truncate
        self._cache_dir = cache_dir
        self._pad_to_max = pad_to_max
        self._bert_dir = bert_dir
        self._vocab_fname = os.path.join(self._bert_dir, 'vocab.txt')
        self._datapath = os.path.join(self._cache_dir, 'pchatkg')
        # NOTE: Loading KG
        # self.entity2index = json.load(open('kgtmp/entity2index.txt',encoding='utf-8'))
        self.entity2index = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in open('kgtmp/entity2index.txt',encoding='utf-8')}
        # self.edge = json.load(open('kgtmp/entity2index.txt',encoding='utf-8'))
        self.stopwords = set([word.strip() for word in open('kgtmp/stopwords.txt',encoding='utf-8')])

    def build_graph_dataset(self, enable_stopwords=False):
        node_num = 12996+1 # FIXME: change to variable
        adj_weight = [{} for i in range(node_num)]
        adj = [[] for i in range(node_num)]
        weight = [[] for i in range(node_num)]
        
        # NOTE: index between 1-12996
        node2index = {line.strip().split('\t')[0]:int(line.strip().split('\t')[1]) for line in open('kgtmp/entity2index.txt',encoding='utf-8')}
        with open('kgtmp/edges.txt',encoding='utf-8')as f:
            edges=set()
            stopwords=set([word.strip() for word in open('kgtmp/stopwords.txt',encoding='utf-8')])
            for line in f:
                lines=line.strip().split('\t')
                entity0=node2index[lines[1].split('/')[0]]
                entity1=node2index[lines[2].split('/')[0]]
                if enable_stopwords and (lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords):
                    continue
                edges.add((entity0,entity1))
                edges.add((entity1,entity0))
            # edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
        
        # XXX: create adj with no weight and no filter(with number of edges)
        for head, tail in edges:
            adj[head].append(tail)
            adj[tail].append(head)

        def get_positive_pairs(node, num_walk=5, walk_depth=2):
            if len(adj[node]) == 0:
                return
            cur_pairs = []
            for i in range(num_walk):
                curr_node = node
                for j in range(walk_depth):
                    neighs = adj[int(curr_node)]
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    # if next_node in self.train_nodes:
                    cur_pairs.append((node, next_node))
                    curr_node = next_node
            return cur_pairs

        def get_negative_pairs(node, num_neg=30, except_depth=2):
            neighbors = set([node])
            frontier = set([node])
            for i in range(except_depth):
                current = set()
                for outer in frontier:
                    current |= set(adj[int(outer)])
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(np.arange(1, len(adj))) - neighbors
            neg_samples = random.sample(list(far_nodes), num_neg) if num_neg < len(far_nodes) else far_nodes
            result = [(node, neg_node) for neg_node in neg_samples]
            return result

        def unsupervise_generator():
            for index, node in enumerate(adj[1:]):
                # FIXME: index+=1
                positive_pairs = get_positive_pairs(index+1)
                negative_pairs = get_negative_pairs(index+1)

                example = {
                    'positive_pairs': positive_pairs,
                    'negative_pairs': negative_pairs
                }
                # example = {
                #     'positive_pairs': tf.constant(list(set(positive_pairs))),
                #     'negative_pairs': tf.constant(list(set(negative_pairs)))
                # }
                yield example

        # XXX: return dataset
        uns_iterator_types = {
            "positive_pairs": tf.int32,
            "negative_pairs": tf.int32,
        }
        unsupervise_dataset = tf.data.Dataset.from_generator(unsupervise_generator, uns_iterator_types)
        # unsupervise_dataset = tf.data.Dataset.from_generator(unsupervise_generator, output_signature=(
        #     tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
        #     tf.TensorSpec(shape=(None, 2), dtype=tf.int32)
        # ))
        # dataset = dataset.shuffle(self._buffer_size).repeat(self._num_epochs) # shuffle
        # colorlog.info(f"Padding parsing")

        def unsupervise_parse_fn(example):# to parallel
            print('wtf')
            print(example['positive_pairs'])
            positive_tensor = tf.constant(example['positive_pairs'], dtype=tf.int32)
            negative_tensor = tf.constant(example['negative_pairs'], dtype=tf.int32)

            example = {
                'positive_pairs': positive_tensor,
                'negative_pairs': negative_tensor
            }
            return example

        # unsupervise_dataset = unsupervise_dataset.map(unsupervise_parse_fn, num_parallel_calls=15)
        return adj, unsupervise_dataset, len(adj[1:])


    @property
    def vocabulary(self) -> data_vocab.Vocabulary:
        if not hasattr(self, '_vocabulary'):
            _vocabulary = data_vocab.Vocabulary(
                vocab_fname=None, vocab_dict=self._dictionary.vocab,
                num_oov_buckets=1, unk_token=data_vocab._BERT_UNK)
            self._vocabulary = _vocabulary
        return self._vocabulary

    def read(self, mode: str, mirrored_strategy: tf.distribute.Strategy = None) -> tf.data.Dataset:
        if mirrored_strategy:
            num_gpus = mirrored_strategy.num_replicas_in_sync
            with mirrored_strategy.scope():
                dataset, num_iters = self._read(mode, self._batch_size * num_gpus)
                dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
            return dataset, num_iters
        else:
            return self._read(mode, self._batch_size)

    def _read(self, mode: str, batch_size: int) -> tf.data.Dataset:
        episodes, dictionary = self._load_and_preprocess_all(mode)
        num_episodes = len(episodes)
        num_examples = sum([len(episode) for episode in episodes])
        num_iters = int(num_episodes / batch_size)

        if mode == 'train':
            self._dictionary = dictionary

        def _gen():
            for episode in episodes:
                examples = {'context': [],
                            'response': [],
                            'chosen_topic': [],
                            'knowledge_sentences': [],
                            'knowledge_kg_masks': []
                            }
                for idx, example in enumerate(episode):
                    if idx == self._max_episode_length:
                        break

                    examples['context'].append(example['context'])
                    examples['response'].append(example['response'])
                    examples['chosen_topic'].append(example['chosen_topic'])

                    if self._knowledge_truncate > 0 and mode == 'train':  # Do not truncate in test time
                        knowledge_sentences = example['knowledge_sentences']
                        knowledge_kg_masks = example['knowledge_kg_masks']
                        num_knowledges = min(len(knowledge_sentences), self._knowledge_truncate)
                        keepers = list(range(1, len(knowledge_sentences)))
                        random.shuffle(keepers)
                        keepers = [0] + keepers[:num_knowledges-1]
                        sentences = itemgetter(*keepers)(knowledge_sentences)
                        kg_masks = itemgetter(*keepers)(knowledge_kg_masks)
                        # 为啥最后append一个大knowledge
                        examples['knowledge_sentences'].append('\n'.join(sentences))
                        m = []
                        for i in kg_masks:
                            m.extend(i)
                        examples['knowledge_kg_masks'].append(m[0])
                    else:
                        knowledge_sentences = example['knowledge_sentences']
                        knowledge_kg_masks = example['knowledge_kg_masks']
                        examples['knowledge_sentences'].append('\n'.join(knowledge_sentences))
                        m = []
                        for i in knowledge_kg_masks:
                            m.extend(i)
                        # examples['knowledge_kg_masks'].append(m)
                        examples['knowledge_kg_masks'].append(m[0])
                        # examples['knowledge_kg_masks'].append('\n'.join(knowledge_kg_masks))

                examples['episode_length'] = len(examples['context'])

                yield examples

        def _parse_fn(example):
            for key, value in self.iterator_shapes.items():
                dims = len(value)
                if dims == len(_scalar()):
                    pass
                elif dims == len(_matrix()):
                    sentences, lengths = list_of_string_split(example[key])
                    example[key] = sentences
                    example[f"{key}_length"] = tf.cast(lengths, tf.int32)
                elif dims == len(_tensor()):
                    list_of_sentences, sentence_lengths, num_sentences = \
                        list_of_list_of_string_split(example[key])
                    if self._max_knowledge > 0:
                        # Truncate length of each knowledge sentences
                        list_of_sentences = list_of_sentences[:, :, :self._max_knowledge]
                        sentence_lengths = tf.minimum(sentence_lengths, self._max_knowledge)
                    example[key] = list_of_sentences
                    example[f"{key}_length"] = tf.cast(sentence_lengths, tf.int32)
                    example[f"num_{key}"] = tf.cast(num_sentences, tf.int32)
                else:
                    raise ValueError

            if self._max_length > 0:
                example['response'] = example['response'][:, :(self._max_length + 1)]
                example['response_length'] = tf.minimum(example['response_length'], self._max_length + 1)
                example['context'] = example['context'][:, :(self._max_length + 1)]
                example['context_length'] = tf.minimum(example['context_length'], self._max_length + 1)

            if self._pad_to_max:
                # XXX : (maybe bug...?) tf.function with dynamic input is extremely slower than
                # static inputs. Therefore, convert dynamic to static.
                episode_max_length = self._max_episode_length
                example['context']                    = tensor_pad(example['context'], [episode_max_length, self._max_length + 1])
                example['response']                   = tensor_pad(example['response'], [episode_max_length, self._max_length + 1])
                example['chosen_topic']               = tensor_pad(example['chosen_topic'], [episode_max_length, 38])
                example['context_length']             = tensor_pad(example['context_length'], [episode_max_length])
                example['response_length']            = tensor_pad(example['response_length'], [episode_max_length])
                example['chosen_topic_length']        = tensor_pad(example['chosen_topic_length'], [episode_max_length])
                if mode == 'train':
                    example['num_knowledge_sentences']    = tensor_pad(example['num_knowledge_sentences'], [episode_max_length])
                    example['knowledge_sentences_length'] = tensor_pad(example['knowledge_sentences_length'], [episode_max_length, self._knowledge_truncate])
                    example['knowledge_sentences']        = tensor_pad(example['knowledge_sentences'], [episode_max_length, self._knowledge_truncate, self._max_knowledge])
                    example['knowledge_kg_masks']        = tensor_pad(example['knowledge_kg_masks'], [episode_max_length, self._knowledge_truncate, self._max_knowledge])
                else:
                    example['num_knowledge_sentences']    = tensor_pad(example['num_knowledge_sentences'], [episode_max_length])
                    example['knowledge_sentences_length'] = tensor_pad(example['knowledge_sentences_length'], [episode_max_length, 175])
                    example['knowledge_sentences']        = tensor_pad(example['knowledge_sentences'], [episode_max_length, 175, self._max_knowledge])
                    example['knowledge_kg_masks']        = tensor_pad(example['knowledge_kg_masks'], [episode_max_length, 175, self._max_knowledge])

            return example

        colorlog.info(f"Construct datagenerator")
        dataset = tf.data.Dataset.from_generator(_gen, self.iterator_types)
        if mode == 'train':
            colorlog.info(f"Shuffle traning set")
            dataset = dataset.shuffle(self._buffer_size).repeat(self._num_epochs)
        else:
            dataset = dataset.repeat(1)
        colorlog.info(f"Padding parsing")
        dataset = dataset.map(_parse_fn, num_parallel_calls=15)

        padded_shapes = {**self.iterator_shapes,
                         'context_length': _vector(),
                         'response_length': _vector(),
                         'chosen_topic_length': _vector(),
                         'knowledge_sentences_length': _matrix(),
                         'num_knowledge_sentences': _vector(),
                         'knowledge_kg_masks_length': _matrix(),
                         'num_knowledge_kg_masks': _vector(),
                         'episode_length': _scalar()}

        drop_remainder = False if mode == 'train' else True
        batched_dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=drop_remainder)

        colorlog.info(f"Reading dataset finished.")
        return batched_dataset, num_iters

    @staticmethod
    def remove_pad(example):
        episode_max_length = tf.reduce_max(example['episode_length'])
        context_length = tf.reduce_max(example['context_length'])
        response_length = tf.reduce_max(example['response_length'])
        topic_length = tf.reduce_max(example['chosen_topic_length'])
        knowledge_length = tf.reduce_max(example['knowledge_sentences_length'])
        num_knowledges = tf.reduce_max(example['num_knowledge_sentences'])

        sliced_example = {}
        sliced_example['episode_length'] = example['episode_length']
        sliced_example['context_length'] = example['context_length'][:, :episode_max_length]
        sliced_example['response_length'] = example['response_length'][:, :episode_max_length]
        sliced_example['chosen_topic_length'] = example['chosen_topic_length'][:, :episode_max_length]
        sliced_example['num_knowledge_sentences'] = example['num_knowledge_sentences'][:, :episode_max_length]
        sliced_example['context'] = example['context'][:, :episode_max_length, :context_length]
        sliced_example['response'] = example['response'][:, :episode_max_length, :response_length]
        sliced_example['chosen_topic'] = example['chosen_topic'][:, :episode_max_length, :topic_length]
        sliced_example['knowledge_sentences_length'] = example['knowledge_sentences_length'][:, :episode_max_length, :num_knowledges]
        sliced_example['knowledge_sentences'] = example['knowledge_sentences'][:, :episode_max_length, :num_knowledges, :knowledge_length]
        sliced_example['knowledge_kg_masks'] = example['knowledge_kg_masks'][:, :episode_max_length, :num_knowledges, :knowledge_length]

        return sliced_example

    def _load_and_preprocess_all(self, mode: str):
        """
        As default, it returns the following action dict:
        {
            'id': 'wizard_of_wikipedia'
            'text': chosen_topic\n # if first example in episode
                    last_apprentice_message\n # if possible
                    wizard_message # if --label-type is 'chosen_sent'
            'knowledge': title_1 sentence_1\n
                                .
                                .
                                .
                         title_m sentence_n # all knowledge available to wizard
            'labels': [title_checked sentence_checked] # default
                                    OR
                      [wizard_response] # if --label-type set to 'response'
            'label_candidates': knowledge + [no_passages_used no_passages_used]
                                           OR
                                100 response candidates  # if 'validation' or 'test'
            'chosen_topic': chosen_topic as untokenized string
            'checked_sentence': checked sentence if wizard, else None # if --include_checked_sentence
            'title': title of checked sentence # if --include_checked_sentence
            --> if not exists, then checked_sentence = title = 'no_passages_used'
            'episode_done': (Boolean) whether episode is done or not
        }
        """
        import os
        if os.path.exists(self._get_preprocessed_fname(mode)):
            episodes_fname = self._get_preprocessed_fname(mode)
            colorlog.info(f"Load cached pchatkg from {episodes_fname}")
            with open(episodes_fname, 'r') as fp:
                episodes = []
                for line in fp:
                    episodes.append(json.loads(line))
            dictionary = tokenization.FullTokenizer(self._vocab_fname)
            return episodes, dictionary

        # parlai_opt = self._get_parlai_opt([
        #     '--task', 'wizard_of_wikipedia:generator:topic_split' if 'unseen' in mode else 'wizard_of_wikipedia:generator:random_split',
        #     '--datatype', '{}:stream'.format(mode.split('_')[0]) if 'unseen' in mode else f'{mode}:stream',  # 'train' for shuffled data and 'train:stream' for unshuffled data
        #     '--datapath', self._cache_dir,
        #     # dict_XXX will not be used if we use bert tokenizer
        #     '--dict_lower', 'True',
        #     '--dict_tokenizer', 'bpe',
        #     '--dict_file', f"{self._cache_dir}/wow.dict",
        #     '--dict_textfields', "text,labels,chosen_topic,checked_sentence,knowledge,title",  # For retrieval mode, use "text,labels"
        #     # By following author's code. For retrieval mode, use 250004
        #     # Also, note that this is the size of bpehelper dictionary.
        #     # So, final dictionary can be larger than this one
        #     # And, don't convert special tokens to index with txt2vec method, you must use tok2ind
        #     '--dict_maxtokens', '30000',
        #     '--dict_nulltoken', data_vocab._PARLAI_PAD,
        #     '--dict_starttoken',data_vocab._PARLAI_GO,
        #     '--dict_endtoken', data_vocab._PARLAI_EOS,
        #     '--dict_unktoken', data_vocab._PARLAI_UNK,
        #     '--include_knowledge_separator', 'True',  # include speical __knowledge__ token between title and passage
        #     '--include_checked_sentence', 'True',
        #     '--label_type', 'response', # choices = ['response', 'chosen_sent']
        # ])
        # # As a default, world use "WizardDialogKnowledgeTeacher"
        # agent = DictionaryAgent(parlai_opt)
        # world = create_task(parlai_opt, agent)
        # num_examples = world.num_examples()
        # num_episodes = world.num_episodes()

        

        # load data
        import os
        from tqdm import tqdm
        g = os.walk('/home/faan/Project/dous/my_DHAP/data30')
        for path, dir_list, file_list in g:
            filenames = file_list
        print('User:', len(filenames))
        
        episodes = []
        
        dictionary = tokenization.FullTokenizer(self._vocab_fname)
        tokenize = lambda x: ' '.join([str(data_vocab.BERT_CLS_ID)] +
            [str(y) for y in dictionary.convert_tokens_to_ids(dictionary.tokenize(x))] + [str(data_vocab.BERT_SEP_ID)])

        # To cmp with kg-preprocessing, to verity our loader.
        count_sent = 0
        count_words = 0
        count_linked_words = 0
        # sentences 30000 user - 60w iter (Overall 79w, train 60 - vaild 10 - test 10)
        for filename in tqdm(filenames[:1000] if mode=='valid' or mode=='test' else filenames[:30000], desc='- Collecting & Tokenizing Sentence'):
            # Collected dict including: text, labels/eval_labels, chosen_topic=null, title=null, checked_sentence
            examples = []
            with open(os.path.join(path, filename), 'r')as f:
                checked_sentences = ['null']
                lines = f.readlines()
                if mode == 'train':
                    lines = lines[:-4]
                elif mode == 'valid':
                    checked_sentences += [i.strip().split('\t')[3] for i in lines[:-4]]
                    lines = lines[-4:-2]
                elif mode == 'test':
                    checked_sentences += [i.strip().split('\t')[3] for i in lines[:-2]]
                    lines = lines[-2:]

                # NOTE: KG masks for checked_sentences
                knowledge_kg_masks = []
                for knowledge in checked_sentences:
                    knowledge_kg_mask = []
                    for word in knowledge.split():
                        # 所有词均进行实体链接，连接失败为0
                        #if word.lower() not in self.stopwords:
                        knowledge_kg_mask.append(self.entity2index.get(word.lower(),'0'))
                        #else:
                        #    concept_mask.append(0)
                    knowledge_kg_masks.append(knowledge_kg_mask)

                checked_sentences = [tokenize(i) for i in checked_sentences]
                
                for line in lines:
                    count_sent += 1
                    example_dict = {}
                    line = line.strip().split('\t')
                    query = tokenize(line[0])
                    resp = tokenize(line[3])
                    example_dict['text'] = query
                    example_dict['labels'] = resp
                    example_dict['knowledge'] = checked_sentences[-5:]
                    example_dict['chosen_topic'] = tokenize('null')
                    example_dict['title'] = 'null'
                    checked_sentences.append(resp)
                    
                    knowledge_kg_mask = []
                    for word in line[3].split():
                        count_words += 1
                        # 所有词均进行实体链接，连接失败为0
                        #if word.lower() not in self.stopwords:
                        knowledge_kg_mask.append(self.entity2index.get(word.lower(),'0'))
                        #else:
                        #    concept_mask.append(0)
                        if knowledge_kg_mask[-1]!='0':
                            count_linked_words += 1
                    example_dict['knowledge_kg_mask'] = knowledge_kg_masks[-5:]
                    knowledge_kg_masks.append(knowledge_kg_mask)
                    
                    examples.append(example_dict)
            
                episodes.append(examples)

        dictionary = tokenization.FullTokenizer(self._vocab_fname)

        # verify
        print(mode, 'linked per sentence:', count_linked_words/count_sent)
        print(mode, 'length per sentence:', count_words/count_sent)

        return self._preprocess_episodes(episodes, dictionary, mode)

    def _get_parlai_opt(self, options: List[str] = [], print_args=False):
        from parlai.scripts.build_dict import setup_args
        parser = setup_args()
        opt = parser.parse_args(options, print_args=print_args)
        return opt

    def _get_preprocessed_fname(self, mode):
        if self._datapath:
            return os.path.join(self._datapath, f'{mode}_episodes.json')
        else:
            return None

    def _preprocess_episodes(self, episodes, dictionary, mode):
        """
        Tokenize all the fields in Wizard-of-Wikipedia
        """
        colorlog.info("Preprocess wizard of wikipedia dataset")
        # tokenize = lambda x: ' '.join([str(data_vocab.BERT_CLS_ID)] +
        #     [str(y) for y in dictionary.convert_tokens_to_ids(dictionary.tokenize(x))] + [str(data_vocab.BERT_SEP_ID)])

        new_episodes = []
        for episode_num, episode in enumerate(tqdm(episodes, ncols=70)):
            new_examples = []
            for example_num, example in enumerate(episode):
                # Tokenize inputs and convert to tokens
                # context = tokenize(example['text'])
                # response = tokenize(example['labels'])
                # chosen_topic = tokenize(example['chosen_topic'])
                context = example['text']
                response = example['labels']
                chosen_topic = example['chosen_topic']

                # Set up knowledge
                # checked_knowledge = example['title'] + ' __knowledge__ ' + example['checked_sentence']
                # knowledges = example['knowledge']
                knowledge_sentences = example['knowledge']
                knowledge_masks = example['knowledge_kg_mask']
                # Tokenize knowledge
                # knowledge_sentences = [tokenize(k) for k in knowledges]

                new_example = {'context': context,
                               'response': response,
                               'chosen_topic': chosen_topic,
                               'knowledge_sentences': knowledge_sentences,
                               'knowledge_kg_masks': knowledge_masks,
                               'episode_num': episode_num,
                               'example_num': example_num}
                new_examples.append(new_example)
            new_episodes.append(new_examples)

        if self._datapath:
            episodes_fname = self._get_preprocessed_fname(mode)
            colorlog.info(f"Cache preprocessed dataset to {episodes_fname}")
            with open(episodes_fname, 'w') as fp:
                for episode in new_episodes:
                    fp.write(json.dumps(episode) + '\n')

        colorlog.info(f"Cached")
        return new_episodes, dictionary
