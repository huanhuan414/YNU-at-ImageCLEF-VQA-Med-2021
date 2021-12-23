from vqa_dataset import VQAFeatureDataset
from tools.create_dictionary import Dictionary
import os
from torch.utils.data import DataLoader
from my_model import BAN_Model
import torch
import _pickle as cPickle
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    # GPU config
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')

    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models',
                        help='save file directory')

    # Training testing or sampling Hyper-parameters
    parser.add_argument('--epochs', type=int, default=550,
                        help='the number of epoches')
    parser.add_argument('--lr', default=0.0007, type=float, metavar='lr',
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')
    parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # # Train with RAD
    parser.add_argument('--use_data', action='store_true', default=True,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--data_dir', type=str,
                        help='RAD dir')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.6, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Attention --------------------------------------------------------------------------------------------------------
    # Choices of attention models
    parser.add_argument('--attention', type=str, default='BAN', choices=['BAN'],
                        help='the model we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--glimpse', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Question ---------------------------------------------------------------------------------------------------------
    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='GRU', choices=['LSTM', 'GRU'],
                        help='the RNN we use')
    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=False,
                        help='tfidf word embedding?')
    parser.add_argument('--cat', type=bool, default=True,
                        help='concatenated 600-D word embedding')
    parser.add_argument('--hid_dim', type=int, default=2000,
                        help='dim of joint semantic features')

    # Vision -----------------------------------------------------------------------------------------------------------
    # Input visual feature dimension
    parser.add_argument('--v_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.003, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')

    # other model hyper-parameters
    parser.add_argument('--other_model', action='store_true', default=True,
                        help='End to end model')

    # details
    parser.add_argument('--details', type=str, default='original ')

    parser.add_argument('--lamda', type=str, default=0.0001)

    args = parser.parse_args()
    return args
def compute_score_with_logits(logits, labels):

    logits = torch.max(logits, 1)[1].data  # argmax

    one_hots = torch.zeros(*labels.size()).to(logits.device)

    one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = (one_hots * labels)

    return scores
def get_keys(d,value):
    return [k for k,v in d.items() if v==value]

if __name__ == '__main__':

    root = os.path.dirname(os.path.abspath(__file__))
    data =root+'/data'
    args = parse_args()
    args.data_dir = data
    # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    d = Dictionary.load_from_file(data + '/dictionary.pkl')
    # prepare the dataloader
    train_dataset = VQAFeatureDataset('train',args,d,dataroot=data)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2,drop_last=False,
                              pin_memory=True)
    validate_dataset = VQAFeatureDataset('validate',args,d,dataroot=data)
    validate_loader = DataLoader(validate_dataset, args.batch_size, shuffle=False, num_workers=2,drop_last=False,
                              pin_memory=True)

    test_dataset = VQAFeatureDataset('test', args, d, dataroot=data)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=2, drop_last=False,
                                 pin_memory=True)

    model = BAN_Model(train_dataset,args)

    checkpoint = torch.load('./saved_models/2021May10-101258/336.pth')

    # checkpoint = torch.load('./saved_models/2021May08-225235/320.pth')


    model.load_state_dict(checkpoint['model_state'])

    model.to(device)

    score = 0
    total = 0


    label_path = data + '/ans2label.pkl'
    if os.path.isfile(label_path):
        print('found %s' % label_path)
        ans2label = cPickle.load(open(label_path, 'rb'))

    label2ans_path = data + '/label2ans.pkl'
    if os.path.isfile(label2ans_path):
        print('found %s' % label2ans_path)
        label2ans = cPickle.load(open(label2ans_path, 'rb'))

    pred_text = './pred.txt'
    model.eval()

    with torch.no_grad():
        for i, (v, q, a, image_name) in enumerate(validate_loader):

            v = v.to(device)

            a = a.to(device)

            preds_close = model(v, q)

            batch_close_score = 0.

            batch_close_score = compute_score_with_logits(preds_close.float(), a.float()).sum()

            score += batch_close_score

            size = v.shape[0]
            total += size  # batch number


            y_pred = torch.max(preds_close, 1)[1].data  # argmax


            # with open(pred_text, 'a') as pred_target:
            #     for idx, index in enumerate(y_pred):
            #         # answers_pred = get_keys(ans2label, index)
            #         answers_pred = label2ans[index]
            #         answers_pred = ''.join(answers_pred)
            #         pred_target.write(image_name[idx] + '|' + answers_pred + "\n")


    score = 100* score / total


    print('[Validate] Val_Acc:{:.6f}%' .format(score))



