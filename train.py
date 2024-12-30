import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import math
from pprint import PrettyPrinter
import random
import numpy as np

import torch  # Torch must be imported before sklearn and tf
import sklearn
import tensorflow as tf
import better_exceptions
from tqdm import tqdm, trange
import colorlog
import colorful

from utils.etc_utils import set_logger, set_tcmalloc, set_gpus, check_none_gradients
from utils import config_utils, custom_argparsers
from models import MODELS
from modules.checkpoint_tracker import CheckpointTracker
from modules.trainer import run_wow_evaluation, Trainer
from modules.from_parlai import download_from_google_drive, unzip
from data.wizard_of_wikipedia import WowDatasetReader
from data.holle import HolleDatasetReader
# from data.pchat import PchatDatasetReader
from data.pchatkg import PchatKGDatasetReader

better_exceptions.hook()
_command_args = config_utils.CommandArgs()
pprint = PrettyPrinter().pprint
pformat = PrettyPrinter().pformat
BEST_N_CHECKPOINTS = 5


def main():
    # Argument passing/parsing
    args, model_args = config_utils.initialize_argparser(MODELS, _command_args, custom_argparsers.DialogArgumentParser)
    hparams, hparams_dict = config_utils.create_or_load_hparams(args, model_args, args.cfg)
    pprint(hparams_dict)

    # Set environment variables & gpus
    set_logger()
    set_gpus(hparams.gpus)
    set_tcmalloc()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Set random seed
    tf.random.set_seed(hparams.random_seed)
    np.random.seed(hparams.random_seed)
    random.seed(hparams.random_seed)

    # For multi-gpu
    if hparams.num_gpus > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()  # NCCL will be used as default
    else:
        mirrored_strategy = None

    # Download BERT pretrained model
    if not os.path.exists(hparams.bert_dir):
        os.makedirs(hparams.bert_dir)
        fname = 'uncased_L-12_H-768_A-12.zip'
        gd_id = '17rfV9CleFBwwfS7m5Yd72vvxdPLWBHl6'
        download_from_google_drive(gd_id, os.path.join(hparams.bert_dir, fname))
        unzip(hparams.bert_dir, fname)

    # Make dataset reader
    os.makedirs(hparams.cache_dir, exist_ok=True)
    if hparams.data_name == "wizard_of_wikipedia":
        reader_cls = WowDatasetReader
    elif hparams.data_name == "holle":
        reader_cls = HolleDatasetReader
    elif hparams.data_name == "pchat":
        reader_cls = PchatKGDatasetReader
    else:
        raise ValueError("data_name must be one of 'wizard_of_wikipedia' and 'holle'")
    reader = reader_cls(
        hparams.batch_size, hparams.num_epochs,
        buffer_size=hparams.buffer_size,
        bucket_width=hparams.bucket_width,
        max_length=hparams.max_length,
        max_episode_length=hparams.max_episode_length,
        max_knowledge=hparams.max_knowledge,
        knowledge_truncate=hparams.knowledge_truncate,
        cache_dir=hparams.cache_dir,
        # bert_dir=hparams.bert_dir,
        bert_dir='bert_pretrained/chinese_L-12_H-768_A-12',
    )
    train_dataset, iters_in_train = reader.read('train', mirrored_strategy)
    test_dataset, iters_in_test = reader.read('valid', mirrored_strategy)
    # test_dataset, iters_in_test = reader.read('test', mirrored_strategy)
    adj, unsupervise_train_dataset, iters_in_uns_train = reader.build_graph_dataset() # get adj_list and dataset
    if hparams.data_name == 'wizard_of_wikipedia':
        unseen_dataset, iters_in_unseen = reader.read('test_unseen', mirrored_strategy)
    vocabulary = reader.vocabulary

    # Build model & optimizer & trainer
    if mirrored_strategy:
        with mirrored_strategy.scope():
            model = MODELS[hparams.model](hparams, vocabulary)
            optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.init_lr,
                                                 clipnorm=hparams.clipnorm)
    else:
        if hparams.model == 'SKT_KG':
            model = MODELS[hparams.model](hparams, vocabulary, adj)
        else:
            model = MODELS[hparams.model](hparams, vocabulary)
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.init_lr,
                                                clipnorm=hparams.clipnorm)
    # trainer = Trainer(model, optimizer, mirrored_strategy,
    #                   hparams.enable_function,
    #                   WowDatasetReader.remove_pad)
    trainer = Trainer(model, optimizer, mirrored_strategy,
                      hparams.enable_function,
                      PchatKGDatasetReader.remove_pad)

    # init weight
    train_example = next(iter(train_dataset))
    _ = trainer.train_step(train_example)

    # misc (tensorboard, checkpoints)
    file_writer = tf.summary.create_file_writer(hparams.checkpoint_dir)
    file_writer.set_as_default()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step_1 = tf.Variable(0, name='unsuper_step')
    global_step_2 = tf.Variable(0, name='main_step')
    init_epoch = 0
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, optimizer_step=global_step)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory=hparams.checkpoint_dir,
                                                    max_to_keep=hparams.max_to_keep)
    checkpoint_tracker = CheckpointTracker(hparams.checkpoint_dir, max_to_keep=BEST_N_CHECKPOINTS)
    if hparams.checkpoint_dir != 'unset':
        if checkpoint_manager.latest_checkpoint is not None:
            # Load
            # train_example = next(iter(train_dataset))
            # _ = trainer.ugly_load_by_single_train_step(train_example)
            # a = model.layers[10].get_weights()
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            with open(os.path.join(hparams.checkpoint_dir, 'sep_step'), 'r')as f:
                line = f.readline()
            s1, s2, init_epoch = line.strip().split('\t')
            global_step_1.assign(tf.Variable(int(s1)))
            global_step_2.assign(tf.Variable(int(s2)))
            init_epoch = int(init_epoch)+1
            # b = model.layers[10].get_weights()
    # print('Load diff:',sum(sum(a[0]!=b[0])))

    # ============================================ Main loop! ============================================
    train_dataset_iter = iter(train_dataset)
    for epoch in range(init_epoch, hparams.num_epochs):
        print(hparams.checkpoint_dir)
        trainer.optimizer.lr.assign(hparams.init_lr*100*pow(0.8,epoch))# stepLR/epochLR, 10epoch-0.1, 20epoch-0.01

        # Graph unsupervise embedding learning
        # base_description = f"(KGE_Train) Epoch {epoch}, GPU {hparams.gpus}"
        for sub_epoch in range(3):
            train_unsupervise_dataset_iter = iter(unsupervise_train_dataset)
            base_description = f"(KGE_Train) Epoch {epoch} - {sub_epoch}"
            train_tqdm = trange(iters_in_uns_train, ncols=120, desc=base_description)
            for current_step in train_tqdm:
                example = next(train_unsupervise_dataset_iter)
                global_step.assign_add(1)
                _global_step = int(global_step)
                global_step_1.assign_add(1)
                _global_step_1 = int(global_step_1)
                # Train
                output_dict = trainer.unsupervise_train_step(example)

                # Print model
                if _global_step == 1:
                    model.print_model()

                # Write results into TF-Board
                loss_str = str(output_dict['loss'].numpy())
                train_tqdm.set_description(f"{base_description}, Unsuper_Loss {loss_str}")
                cur_lr = trainer.optimizer.lr
                with file_writer.as_default():
                    if _global_step % int(hparams.logging_step) == 0:
                        # tf.summary.histogram('train/vocab', output_dict['sample_ids'], step=_global_step)
                        tf.summary.scalar('global_train/loss', output_dict['loss'], step=_global_step)
                        tf.summary.scalar('global_train/lr', cur_lr, step=_global_step)
                        tf.summary.scalar('kge_train/loss', output_dict['loss'], step=_global_step_1)
                        # tf.summary.scalar('train/gen_loss', output_dict['gen_loss'], step=_global_step)
                        # tf.summary.scalar('train/knowledge_loss', output_dict['knowledge_loss'], step=_global_step)
                        # tf.summary.scalar('train/kl_loss', output_dict['kl_loss'], step=_global_step)


        trainer.optimizer.lr.assign(hparams.init_lr*pow(0.98,epoch))# pow(0.98,epoch) # 20epoch-0.66, # 0.95^40 - 0.128

        # NLG training
        # base_description = f"(Train) Epoch {epoch}, GPU {hparams.gpus}"
        base_description = f"(Train) Epoch {epoch}"
        train_tqdm = trange(iters_in_train, ncols=120, desc=base_description) # ncols=120
        for current_step in train_tqdm:
            example = next(train_dataset_iter)
            global_step.assign_add(1)
            _global_step = int(global_step)
            global_step_2.assign_add(1)
            _global_step_2 = int(global_step_2)

            # Train
            output_dict = trainer.train_step(example)

            # # Print model
            # if _global_step == 1:
            #     model.print_model()

            loss_str = str(output_dict['loss'].numpy())
            train_tqdm.set_description(f"{base_description}, Loss {loss_str}")
            cur_lr = trainer.optimizer.lr
            with file_writer.as_default():
                if _global_step % int(hparams.logging_step) == 0:
                    tf.summary.histogram('train/vocab', output_dict['sample_ids'], step=_global_step_2)
                    tf.summary.scalar('train/loss', output_dict['loss'], step=_global_step_2)
                    tf.summary.scalar('global_train/loss', output_dict['loss'], step=_global_step)
                    tf.summary.scalar('global_train/lr', cur_lr, step=_global_step)
                    tf.summary.scalar('train/gen_loss', output_dict['gen_loss'], step=_global_step_2)
                    tf.summary.scalar('train/knowledge_loss', output_dict['knowledge_loss'], step=_global_step_2)
                    tf.summary.scalar('train/kl_loss', output_dict['kl_loss'], step=_global_step_2)

            # Test per epoch
            if _global_step_2 % int(iters_in_train * hparams.evaluation_epoch) == 0:
                checkpoint_manager.save(global_step)
                
                # save steps, separately
                with open(os.path.join(hparams.checkpoint_dir, 'sep_step'), 'w')as f:
                    f.write(str(_global_step_1)+'\t'+str(_global_step_2)+'\t'+str(epoch))

                test_loop_outputs = trainer.test_loop(test_dataset, iters_in_test, epoch, 'seen')
                if hparams.data_name == 'wizard_of_wikipedia':
                    unseen_loop_outputs = trainer.test_loop(unseen_dataset, iters_in_unseen, epoch, 'unseen')

                test_summaries, log_dict = run_wow_evaluation(
                    test_loop_outputs, os.path.join(hparams.checkpoint_dir, 'ckpt-'+str(_global_step)), 'test')
                    # test_loop_outputs, hparams.checkpoint_dir, 'seen')
                if hparams.data_name == 'wizard_of_wikipedia':
                    unseen_summaries, unseen_log_dict = run_wow_evaluation(
                        unseen_loop_outputs, hparams.checkpoint_dir, 'unseen')

                # Logging
                tqdm.write(colorful.bold_green("seen").styled_string)
                tqdm.write(colorful.bold_red(pformat(log_dict)).styled_string)
                if hparams.data_name == 'wizard_of_wikipedia':
                    tqdm.write(colorful.bold_green("unseen").styled_string)
                    tqdm.write(colorful.bold_red(pformat(unseen_log_dict)).styled_string)

                with file_writer.as_default():
                    for family, test_summary in test_summaries.items():
                        for key, value in test_summary.items():
                            tf.summary.scalar(f'{family}/{key}', value, step=_global_step)
                    if hparams.data_name == 'wizard_of_wikipedia':
                        for family, unseen_summary in unseen_summaries.items():
                            for key, value in unseen_summary.items():
                                tf.summary.scalar(f'{family}/{key}', value, step=_global_step)

                if hparams.keep_best_checkpoint:
                    current_score = log_dict["rouge1"]
                    checkpoint_tracker.update(current_score, _global_step)

if __name__ == '__main__':
    main()
