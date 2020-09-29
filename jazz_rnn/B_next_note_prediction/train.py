import argparse
import time
import os
import logging.config
import sys

from jazz_rnn.utilspy.log import ResultsLog, setup_logging
from jazz_rnn.utilspy.meters import AverageMeter, accuracy
from jazz_rnn.B_next_note_prediction.model import *
from jazz_rnn.utils.music.MusicCorpus import create_music_corpus


def train(args):
    parser = argparse.ArgumentParser(description='PyTorch Jazz RNN/LSTM Model')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', type=str, default='LSTM',
                              help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    model_parser.add_argument('--pitch_emsize', type=int, default=256,
                              help='size of pitch embeddings')
    model_parser.add_argument('--dur_emsize', type=int, default=256,
                              help='size of duration embeddings')
    model_parser.add_argument('--hidden_size', type=int, default=512,
                              help='number of hidden units per layer')
    model_parser.add_argument('--num_layers', type=int, default=3,
                              help='number of layers')
    model_parser.add_argument('--lr', type=float, default=0.5,
                              help='initial learning rate')
    model_parser.add_argument('--clip', type=float, default=0.25,
                              help='gradient clipping')
    model_parser.add_argument('--wd', type=float, default=1e-6,
                              help='weight decay')
    model_parser.add_argument('--wdrop', type=float, default=0.0,
                              help='weight drop applied to hidden-to-hidden weights (0 = no dropout)')
    model_parser.add_argument('--dropouti', type=float, default=0.0,
                              help='dropout applied to embedding (0 = no dropout)')
    model_parser.add_argument('--dropoute', type=float, default=0.0,
                              help='dropout applied to whole embeddings (0 = no dropout)')
    model_parser.add_argument('--dropouth', type=float, default=0.0,
                              help='dropout applied to between rnn layers (0 = no dropout)')
    model_parser.add_argument('--not-tied', dest='tied', action='store_false', default='store_true',
                              help='tie the word embedding and softmax weights')
    model_parser.add_argument('--model-type', default='chord_pitches', choices=rnn_models_dict.keys(),
                              help='which model to use')
    model_parser.add_argument('--normalize', action='store_true',
                              help='normalize word_embedding and rnn output')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--data-pkl', type=str, default='../A_data_prep/results/dataset_pkls/',
                                 help='location of the pickled data corpus')
    training_parser.add_argument('--epochs', type=int, default=500,
                                 help='upper epoch limit')
    training_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                                 help='batch size')
    training_parser.add_argument('--bptt', type=int, default=100,
                                 help='sequence length')
    training_parser.add_argument('--seed', type=int, default=1111,
                                 help='random seed')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='don''t use CUDA')
    training_parser.add_argument('--all', action='store_true', default=False,
                                 help='train using train+test data')
    training_parser.add_argument('--resume', type=str, default='',
                                 help='pkl path to load')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                                help='report interval')
    logging_parser.add_argument('--save', type=str, default='model_' + time.strftime("%y_%m_%d"),
                                help='model name')
    logging_parser.add_argument('--save-dir', type=str, default='results/training_results',
                                help='path to save the final model')

    args = parser.parse_args(args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably run without --no-cuda")
        else:
            torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    save_path = os.path.join(args.save_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_path = os.path.join(save_path, args.save + '.pt')
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to {}".format(checkpoint_path))
    logging.debug("run arguments: %s", args)

    meters_list = ['p_loss', 'd_loss',
                   'p_top1', 'p_top3', 'p_top5',
                   'd_top1', 'd_top3',
                   ]
    LINE_LENGTH = 160

    ###############################################################################
    # Load data
    ###############################################################################
    train_set, val_set, converter = create_music_corpus(pkl_path=args.data_pkl, sequence_len=args.bptt,
                                                        cuda=(not args.no_cuda), all=args.all,
                                                        batch_size=args.batch_size)
    os.system('cp ' + os.path.join(args.data_pkl, '*.pkl') + ' ' + save_path)

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.resume:
        with open(args.resume, 'rb') as f:
            model = torch.load(f)
            # added for backward compatibility
            model.normalize = args.normalize
    else:
        model = rnn_models_dict[args.model_type](
            pitch_sizes=(129, args.pitch_emsize),
            duration_sizes=(train_set.converter.max_durations(), args.dur_emsize),
            root_sizes=(13, 16),
            offset_sizes=(48, 16),
            scale_sizes=(13, 16),
            chord_sizes=(13, 16),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            tie_weights=args.tied,
            wdrop=args.wdrop, dropouti=args.dropouti,
            dropouth=args.dropouth, dropoute=args.dropoute)
    if not args.no_cuda:
        model.cuda()

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()

    logging.info('num_params: {}'.format(num_params))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400, 450], gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Training code
    ###############################################################################

    def evaluate(val_set):
        losses = AverageMeter()
        meters_avg = {k: AverageMeter() for k in meters_list}
        val_batch_size = args.batch_size

        # Turn on evaluation mode which disables dropout
        model.eval()

        hidden = model.init_hidden(val_batch_size)
        num_batches = val_set.get_num_batches()
        with torch.no_grad():
            for i in range(num_batches):
                input, target = val_set.get_batch(i)
                target = target.view(val_batch_size * args.bptt, -1)

                output_pitch, output_duration, hidden = model(input, hidden)

                loss, meters = get_music_loss(output_pitch, output_duration, target)

                losses.update(float(loss), args.bptt)
                for k, v in meters.items():
                    meters_avg[k].update(float(meters[k]), args.bptt)

                hidden = model.init_hidden(args.batch_size)

        for k in meters_avg:
            meters_avg[k] = meters_avg[k].avg
        return losses.avg, meters_avg

    def train(train_set):
        # Turn on training mode which enables dropout.
        model.train()
        losses = AverageMeter()
        errors_avg = {k: AverageMeter() for k in meters_list}
        errors_logging = {k: AverageMeter() for k in meters_list}
        logging_loss = AverageMeter()
        start_time = time.time()
        num_batches = train_set.get_num_batches()
        for i in range(num_batches):

            input, target = train_set.get_random_batch()
            target = target.view(args.batch_size * args.bptt, -1)

            hidden = model.init_hidden(args.batch_size)
            model.zero_grad()

            output_pitch, output_duration, hidden = model(input, hidden)
            loss, errors = get_music_loss(output_pitch, output_duration, target)

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            if args.normalize:
                model.norm_emb()

            losses.update(float(loss), input.size(0))
            logging_loss.update(float(loss), input.size(0))
            for k, v in errors.items():
                errors_avg[k].update(float(errors[k]), args.bptt)
                errors_logging[k].update(float(errors[k]), args.bptt)

            if i % args.log_interval == 0 and i > 0:
                cur_loss = logging_loss.avg
                elapsed = time.time() - start_time
                error_str = ' | '.join(['{}: {:5.2f}'.format(k, v.avg) for k, v in errors_logging.items()])

                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                             'loss {:5.2f} | ppl {:8.2f} | grad_norm {:5.2f} | {}'
                             .format(epoch, i, num_batches,
                                     elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), grad_norm,
                                     error_str))
                logging_loss.reset()
                for k, v in errors.items():
                    errors_avg[k].reset()
                start_time = time.time()

        for k in errors_avg:
            errors_avg[k] = errors_avg[k].avg
        return losses.avg, errors_avg

    def get_music_loss(pitch_logits, duration_logits, target_var):
        pitch_loss = criterion(pitch_logits, target_var[:, 0])
        duration_loss = criterion(duration_logits, target_var[:, 1])

        pitch_top1, pitch_top3, pitch_top5 = accuracy(pitch_logits, target_var[:, 0].contiguous(), topk=(1, 3, 5))
        duration_top1, duration_top3 = accuracy(duration_logits, target_var[:, 1].contiguous(), topk=(1, 3))

        meters = {'p_loss': pitch_loss, 'd_loss': duration_loss,
                  'p_top1': pitch_top1, 'p_top3': pitch_top3, 'p_top5': pitch_top5,
                  'd_top1': duration_top1, 'd_top3': duration_top3,
                  }

        return pitch_loss + duration_loss, meters

    best_val_error = None
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss, train_errors = train(train_set)
            if not args.all:
                val_loss, val_errors = evaluate(val_set)

                logging.info('-' * LINE_LENGTH)
                error_str = ' | '.join(['valid_{}: {:5.2f}'.format(k, v) for k, v in val_errors.items()])
                logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                             'valid ppl {:8.2f} | {}'
                             .format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss), error_str))
                logging.info('-' * LINE_LENGTH)
                # Save the model if the validation loss is the best we've seen so far.
                avg_val_error = (val_errors['p_top1'] + val_errors['d_top1']) / 2
                if not best_val_error or avg_val_error < best_val_error:
                    with open(checkpoint_path.replace('.pt', '_best_val.pt'), 'wb') as f:
                        torch.save(model, f)
                        best_val_error = avg_val_error

                if epoch % 50 == 0:
                    with open(checkpoint_path.replace('.pt', '_e{}.pt'.format(epoch)), 'wb') as f:
                        torch.save(model, f)
                        best_val_error = avg_val_error

                results.add(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            **{'train_{}'.format(k): v for k, v in train_errors.items()},
                            **{'val_{}'.format(k): v for k, v in val_errors.items()})

                results.plot(x='epoch', y=['train_loss', 'val_loss'],
                             title='Loss', ylabel='loss')
                for k in train_errors:
                    results.plot(x='epoch', y=['train_{}'.format(k), 'val_{}'.format(k)],
                                 title=k, ylabel=k)
                results.save()
            scheduler.step()

        logging.info('=' * LINE_LENGTH)
        if args.save != '' or args.all:
            with open(checkpoint_path, 'wb') as f:
                torch.save(model, f)
        return val_errors['p_top1']

    except KeyboardInterrupt:
        print('-' * LINE_LENGTH)
        print('Exiting from training early')

        logging.info('=' * LINE_LENGTH)
        if args.save != '' or args.all:
            checkpoint_path = os.path.join(save_path, args.save + '_early_stop' + '.pt')
            with open(checkpoint_path, 'wb') as f:
                torch.save(model, f)
        return val_errors['p_top1']


if __name__ == '__main__':
    train(sys.argv[1:])
