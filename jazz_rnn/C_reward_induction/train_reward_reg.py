import argparse
import time
import os
import logging.config
import sys
import math

import numpy as np
import torch
import torch.nn as nn

from jazz_rnn.utils.utils import WeightDrop
from jazz_rnn.utilspy.log import ResultsLog, setup_logging
from jazz_rnn.utilspy.meters import AverageMeter, accuracy
from jazz_rnn.C_reward_induction.RewardMusicCorpusReg import RewardMusicCorpus, create_music_corpus

torch.backends.cudnn.enabled = False


class TrainReward:
    def __init__(self, args):
        parser = argparse.ArgumentParser(description='PyTorch Jazz RNN/LSTM Model')

        model_parser = parser.add_argument_group('Model Parameters')
        model_parser.add_argument('--pretrained_path', type=str,
                                  help='path to pre-trained model. used to extract embeddings')
        model_parser.add_argument('--pitch_emsize', type=int, default=256,
                                  help='size of pitch embeddings')
        model_parser.add_argument('--dur_emsize', type=int, default=256,
                                  help='size of duration embeddings')
        model_parser.add_argument('--hidden_size', type=int, default=512,
                                  help='number of hidden units per layer')
        model_parser.add_argument('--n_classes', type=int, default=3,
                                  help='size of network output (n_classes)')
        model_parser.add_argument('--num_layers', type=int, default=3,
                                  help='number of layers')
        model_parser.add_argument('--lr', type=float, default=0.1,
                                  help='initial learning rate')
        model_parser.add_argument('--clip', type=float, default=0.25,
                                  help='gradient clipping')
        model_parser.add_argument('--wd', type=float, default=1e-6,
                                  help='weight decay')
        model_parser.add_argument('--wdrop', type=float, default=0.8,
                                  help='weight drop applied to hidden-to-hidden weights (0 = no dropout)')
        model_parser.add_argument('--dropouti', type=float, default=0.0,
                                  help='dropout applied to embedding (0 = no dropout)')
        model_parser.add_argument('--dropoute', type=float, default=0.0,
                                  help='dropout applied to whole embeddings (0 = no dropout)')
        model_parser.add_argument('--dropouth', type=float, default=0.8,
                                  help='dropout applied to between rnn layers (0 = no dropout)')
        model_parser.add_argument('--dropouta', type=float, default=0.8,
                                  help='dropout applied to attention layers (0 = no dropout)')
        model_parser.add_argument('--normalize', action='store_true',
                                  help='normalize word_embedding and rnn output')

        training_parser = parser.add_argument_group('Training Parameters')
        training_parser.add_argument('--data-pkl', type=str,
                                     help='location of the pickled data corpus', required=True)
        training_parser.add_argument('--epochs', type=int, default=100,
                                     help='upper epoch limit')
        training_parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                                     help='batch size')
        training_parser.add_argument('--bptt', type=int, default=16,
                                     help='sequence length')
        training_parser.add_argument('--seed', type=int, default=1111,
                                     help='random seed')
        training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                     help='don''t use CUDA')
        training_parser.add_argument('--all', action='store_true', default=False,
                                     help='train using train+test data')
        training_parser.add_argument('--test', action='store_true', default=False,
                                     help='perform test only')
        training_parser.add_argument('--resume', type=str, default='',
                                     help='pkl path to load')
        training_parser.add_argument('--train-chord-root', action='store_true', default=False,
                                     help='train model to play only chord root')

        logging_parser = parser.add_argument_group('Logging Parameters')
        logging_parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                                    help='report interval')
        logging_parser.add_argument('--save', type=str,
                                    default='model_user' + time.strftime("%y_%m_%d"),
                                    help='model name')
        logging_parser.add_argument('--save-dir', type=str, default='results/reward_training_results',
                                    help='path to save the final model')
        self.args = parser.parse_args(args)
        self.args.cuda = not self.args.no_cuda

        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if self.args.no_cuda:
                logging.info("WARNING: You have a CUDA device, so you should probably run without --no-cuda")
            else:
                torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)

        self.save_path = os.path.join(self.args.save_dir, self.args.save)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        setup_logging(os.path.join(self.save_path, 'log.txt'))
        results_file = os.path.join(self.save_path, 'results.%s')
        self.results = ResultsLog(results_file % 'csv', results_file % 'html', epochs=self.args.epochs)

        logging.debug("run arguments: %s", self.args)

        per_class_accuracies = ['acc_{}'.format(i) for i in range(self.args.n_classes)]
        self.meters_list = ['loss', 'acc', *per_class_accuracies
                            ]

        ###############################################################################
        # Load data
        ###############################################################################
        self.all_data_list, self.idxs_kf, self.converter, _ = create_music_corpus(pkl_path=self.args.data_pkl,
                                                                                  all=self.args.all)

        os.system('cp ' + os.path.join(self.args.data_pkl, '*.pkl') + ' ' + self.save_path)

    def evaluate(self, val_set, epoch):
        losses = AverageMeter()
        meters_avg = {k: AverageMeter() for k in self.meters_list}
        val_batch_size = self.args.batch_size

        # Turn on evaluation mode which disables dropout
        self.model.eval()
        hidden = self.model.init_hidden(val_batch_size)

        num_batches = val_set.get_num_batches()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i in range(num_batches):
                input, target = val_set.get_batch(i)
                target = target.view(val_batch_size, -1)

                output_reward, hidden = self.model.forward_reward(input, hidden)

                loss, meters = self.get_music_loss(output_reward, target)

                losses.update(float(loss), self.args.bptt)
                for k, v in meters.items():
                    meters_avg[k].update(float(meters[k]), self.args.bptt)

                y_true.append(target.view(1, val_set.batch_size).mode(dim=0)[0].cpu().numpy())
                y_pred.append(output_reward.argmax(dim=1).cpu().numpy())

        for k in meters_avg:
            meters_avg[k] = meters_avg[k].avg

        return losses.avg, meters_avg

    def train(self, train_set, epoch):
        # Turn on training mode which enables dropout.
        self.model.train()
        losses = AverageMeter()
        errors_avg = {k: AverageMeter() for k in self.meters_list}
        errors_logging = {k: AverageMeter() for k in self.meters_list}
        logging_loss = AverageMeter()
        start_time = time.time()
        train_set.balance_dataset()
        num_batches = train_set.get_num_batches()
        train_set.permute_order()
        indices = np.random.permutation(np.arange(num_batches))
        for i in range(len(indices)):
            input, target = train_set.get_batch(i)
            target = target.view(self.args.batch_size, -1)

            hidden = self.model.init_hidden(self.args.batch_size)
            self.model.zero_grad()
            output_reward, hidden = self.model.forward_reward(input, hidden)

            loss, errors = self.get_music_loss(output_reward, target)

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()

            losses.update(float(loss), target.size(0))
            logging_loss.update(float(loss), target.size(0))
            for k, v in errors.items():
                errors_avg[k].update(float(errors[k]), self.args.bptt)
                errors_logging[k].update(float(errors[k]), self.args.bptt)

            if i % self.args.log_interval == 0 and i > 0:
                cur_loss = logging_loss.avg
                elapsed = time.time() - start_time
                error_str = ' | '.join(['{}: {:5.2f}'.format(k, v.avg) for k, v in errors_logging.items()])

                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                             'loss {:5.2f} | ppl {:8.2f} | grad_norm {:5.2f} | {}'
                             .format(epoch, i, num_batches,
                                     elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss), grad_norm,
                                     error_str))
                logging_loss.reset()
                for k, v in errors.items():
                    errors_logging[k].reset()
                start_time = time.time()

        for k in errors_avg:
            errors_avg[k] = errors_avg[k].avg
        return losses.avg, errors_avg

    TOTAL_NOTES_IN_MIDI = 128
    N_NOTES_IN_OCTAVE = 12
    RESIDUAL = TOTAL_NOTES_IN_MIDI % N_NOTES_IN_OCTAVE

    def get_music_loss(self, out_logits, target_var):
        loss = self.criterion(torch.tanh(out_logits), target_var)
        loss = torch.mean(loss)

        try:
            if len(torch.unique(target_var)) < 3:
                raise IndexError('can''t calculate per class acc')
        except IndexError:
            return loss, {}

        acc = accuracy(torch.sign(out_logits), torch.sign(target_var.contiguous()))[0]
        per_class_accuracies = {}
        meters = {'loss': loss, 'acc': acc, **per_class_accuracies}

        return loss, meters

    def main(self):

        for self.fold_idx, (train_index, val_index) in enumerate(self.idxs_kf):
            train_data_fold = np.array(self.all_data_list)[train_index]
            logging.info('Training set statistics: ')
            logging.info(np.histogram(np.concatenate(train_data_fold, axis=0)[:, -1], bins=3)[0])

            self.train_corpus = RewardMusicCorpus(train_data_fold, self.converter, cuda=self.args.cuda,
                                                  batch_size=self.args.batch_size, balance=True,
                                                  n_classes=self.args.n_classes, seq_len=self.args.bptt)

            if not self.args.all:
                val_data_fold = np.array(self.all_data_list)[val_index]
                logging.info('Validation set statistics: ')
                logging.info(np.histogram(np.concatenate(val_data_fold, axis=0)[:, -1], bins=3)[0])

                self.val_corpus = RewardMusicCorpus(val_data_fold, self.converter, cuda=self.args.cuda,
                                                    batch_size=self.args.batch_size, balance=False,
                                                    n_classes=self.args.n_classes, seq_len=self.args.bptt)

            self.checkpoint_path = os.path.join(self.save_path, self.args.save + '_f{}.pt'.format(self.fold_idx))
            ###############################################################################
            # Build the model
            ###############################################################################

            if self.args.resume:
                with open(self.args.resume, 'rb') as f:
                    self.model = torch.load(f)
            if self.args.pretrained_path:
                with open(self.args.pretrained_path, 'rb') as f:
                    self.model = torch.load(f)
                    # added for backward compatibility
                    self.model.normalize = self.args.normalize

            for p in [self.model.encode_pitch, self.model.encode_duration, self.model.encode_offset,
                      self.model.decode_pitch, self.model.decode_duration]:
                p.requires_grad = False
            if isinstance(self.model.rnns[0], nn.LSTM) and self.args.wdrop != 0:
                self.model.rnns = nn.ModuleList(
                    [WeightDrop(rnn, ['weight_hh_l0'], dropout=self.args.wdrop) for rnn in self.model.rnns])
            self.model.dropoute = self.args.dropoute
            self.model.dropouth = self.args.dropouth
            self.model.dropouti = self.args.dropouti
            self.model.dropouta = nn.Dropout(self.args.dropouta)
            self.model.wdrop = self.args.wdrop

            self.model.reward_attention = nn.Linear(1552 + 512, 1)
            self.model.reward_linear = nn.Linear(512, 1)

            if not self.args.no_cuda:
                self.model.cuda()

            num_params = 0
            for p in self.model.parameters():
                num_params += p.numel()
            logging.info('num_params: {}'.format(num_params))

            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=self.args.lr,
                                             momentum=0.9, weight_decay=self.args.wd, nesterov=True)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150, 300, 400],
                                                                  gamma=0.5)

            self.criterion = nn.MSELoss()

            LINE_LENGTH = 160

            epoch_start_time = time.time()
            if not self.args.all:
                val_loss, val_errors = self.evaluate(self.val_corpus, 0)
                self.results.add(epoch=0,
                                 **{'f{}_val_{}'.format(self.fold_idx, k): v for k, v in val_errors.items()})
                logging.info('-' * LINE_LENGTH)
                logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                             'valid ppl {:8.2f}'.format(0, (time.time() - epoch_start_time),
                                                        val_loss, math.exp(val_loss)))
                logging.info('-' * LINE_LENGTH)

            # Loop over epochs.
            best_val_error = None
            # At any point you can hit Ctrl + C to break out of training early.
            if self.args.test:
                val_loss, val_errors = self.evaluate(self.val_corpus, 0)
                logging.info('-' * LINE_LENGTH)
                error_str = ' | '.join(['valid_{}: {:5.2f}'.format(k, v) for k, v in val_errors.items()])
                logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                             'valid ppl {:8.2f} | {}'
                             .format(0, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss), 0, error_str))
                logging.info('-' * LINE_LENGTH)
                return val_errors['acc']
            try:
                for epoch in range(1, self.args.epochs + 1):
                    epoch_start_time = time.time()
                    train_loss, train_errors = self.train(self.train_corpus, epoch)
                    if not self.args.all:
                        val_loss, val_errors = self.evaluate(self.val_corpus, epoch)

                        logging.info('-' * LINE_LENGTH)
                        error_str = ' | '.join(['valid_{}: {:5.2f}'.format(k, v) for k, v in val_errors.items()])
                        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                     'valid ppl {:8.2f} | {}'
                                     .format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss), error_str))
                        logging.info('-' * LINE_LENGTH)
                        # Save the model if the validation loss is the best we've seen so far.
                        avg_val_error = val_errors['acc']
                        if not best_val_error or avg_val_error < best_val_error:
                            with open(self.checkpoint_path, 'wb') as f:
                                torch.save(self.model, f)
                                best_val_error = avg_val_error

                        self.results.add(epoch=epoch,
                                         **{'f{}_train_{}'.format(self.fold_idx, k): v for k, v in
                                            train_errors.items()},
                                         **{'f{}_val_{}'.format(self.fold_idx, k): v for k, v in val_errors.items()})

                        for k in train_errors:
                            self.results.plot(x='epoch', y=['train_{}'.format(k), 'val_{}'.format(k)],
                                              title=k, ylabel=k, avg=True)
                        self.results.save()

                        if (self.args.save != '' and (epoch % 10) == 0) or self.args.all:
                            with open(self.checkpoint_path, 'wb') as f:
                                torch.save(self.model, f)

                    self.scheduler.step()

                logging.info('=' * LINE_LENGTH)
                if self.args.save != '' or self.args.all:
                    with open(self.checkpoint_path, 'wb') as f:
                        torch.save(self.model, f)

            except KeyboardInterrupt:
                logging.info('-' * LINE_LENGTH)
                logging.info('Exiting from training early')

                logging.info('=' * LINE_LENGTH)
                if self.args.save != '' or self.args.all:
                    self.checkpoint_path = os.path.join(self.save_path, self.args.save + '_early_stop' + '.pt')
                    with open(self.checkpoint_path, 'wb') as f:
                        torch.save(self.model, f)


if __name__ == '__main__':
    TrainReward(sys.argv[1:]).main()
