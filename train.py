import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import traceback


class ConvBlock(nn.Module):
    def __init__(self, *args, batch_norm=False, dropout=0, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(args[1])
        else:
            self.batch_norm = False

    def forward(self, inputs):
        if self.batch_norm:
            return F.leaky_relu(self.dropout(self.batch_norm(self.conv(inputs))))
        else:
            return F.leaky_relu(self.dropout(self.conv(inputs)))


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.conv1   = ConvBlock(5, 64, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv2   = ConvBlock(64, 64, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv3   = ConvBlock(64, 128, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv4   = ConvBlock(128, 128, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv5   = ConvBlock(128, 256, 3, batch_norm=True, dropout=.2, stride=2)
        self.conv6   = ConvBlock(256, 256, 3, batch_norm=True, dropout=.3, stride=1)
        self.conv7   = ConvBlock(256, 512, 3, batch_norm=True, dropout=.4, stride=2)
        self.conv8   = ConvBlock(512, 512, 3, batch_norm=True, dropout=.5, stride=1)
        self.conv9   = ConvBlock(512, 1, 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x.view(x.size(0), -1)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        predictions = [model(inputs) for model in self.models]
        shape = predictions[0].shape
        predictions = torch.cat([pred.view(shape[0], -1, 1) for pred in predictions], dim=2)
        mean_prediction = torch.mean(predictions, dim=2).view(shape)
        return mean_prediction


def save_losses(losses, destination):
    colors = ['red', 'blue', 'yellow', 'black', 'green', 'magenta', 'cyan']
    markers = ['o', 's', '*', '+', 'D', '|', '_']
    losses = losses.view(losses.size(0), losses.size(1), -1)
    torch.save(losses.detach(), destination+'.pt')
    losses = losses.numpy()

    for i in range(losses.shape[0]):
        plt.plot(range(losses.shape[1]), losses[i, :, 0], color=colors[i], marker=markers[i], linestyle='solid')
        if losses.shape[2] == 2:
            plt.plot(range(losses.shape[1]), losses[i, :, 1], color=colors[i], marker=markers[i], linestyle='dashed')

    plt.savefig(destination+'.png')
    plt.clf()


def build_model(args):
    return ConvolutionalModel().cuda()


def shuffle_data(inputs, outputs):
    i_shuffle = torch.randperm(inputs.size(0))
    return inputs[i_shuffle], outputs[i_shuffle]


def step(model, inputs, outputs, loss_f, opt):
    predictions = model(inputs).view(-1)
    loss = loss_f(predictions, outputs)
    torch.mean(loss).backward()
    opt.step()
    opt.zero_grad()
    return loss


def train(model, inputs, outputs, args):
    bs = args.batch_size
    inputs, outputs = shuffle_data(inputs, outputs)
    test_inputs = inputs[:inputs.size(0)//100]
    test_outputs = outputs[:outputs.size(0)//100]
    inputs = inputs[inputs.size(0)//100:]
    outputs = outputs[outputs.size(0)//100:]
    data_size = inputs.size(0)
    partition_size = data_size // args.partitions
    loss_f = nn.MSELoss(reduction='none')
    eval_f = nn.MSELoss(reduction='sum')

    ''' K-fold Cross Validation'''
    cv_losses = torch.zeros(args.partitions, args.epochs, 2)
    fold_models = [copy.deepcopy(model) for i in range(args.partitions)]
    fold_opts = [optim.Adam(model.parameters(), lr=args.learning_rate) for model in fold_models]
    mean_loss = 0
    for i_fold in range(args.partitions):
        args.batch_size = bs
        model = fold_models[i_fold]
        opt = fold_opts[i_fold]
        eval_inputs = inputs[i_fold*partition_size:(i_fold+1)*partition_size]
        eval_outputs = outputs[i_fold*partition_size:(i_fold+1)*partition_size]
        train_inputs = torch.cat([inputs[:i_fold*partition_size], inputs[(i_fold+1)*partition_size:]])
        train_outputs = torch.cat([outputs[:i_fold*partition_size], outputs[(i_fold+1)*partition_size:]])
        data_mean = torch.mean(eval_outputs).detach()
        data_error = eval_f(eval_outputs.detach(), torch.ones(eval_outputs.size(0))*data_mean).detach() / eval_outputs.size(0)

        n_batches = (train_inputs.size(0) // args.batch_size)+1
        train_losses = torch.zeros(n_batches)
        if args.loss_sampling:
            logits = torch.zeros(train_inputs.size(0))
        for i_epoch in range(args.epochs):
            args.batch_size = int(args.batch_size * args.batch_size_annealing)

            # Train Epoch
            model.train()
            train_inputs, train_outputs = shuffle_data(train_inputs, train_outputs)
            for i_batch in range(n_batches):
                if args.loss_sampling and i_epoch > 0:
                    batch_indices = torch.multinomial(logits, args.batch_size, replacement=True)
                else:
                    batch_indices = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)

                batch_inputs = train_inputs[batch_indices].cuda()
                batch_outputs = train_outputs[batch_indices].cuda()
                if batch_inputs.size(0) == 0:
                    continue
                batch_losses = step(model, batch_inputs, batch_outputs, loss_f, opt)
                train_losses[i_batch] = torch.mean(batch_losses).detach()

                if args.loss_sampling:
                    with torch.no_grad():
                        logits[batch_indices] = batch_losses.cpu()

            # Eval Epoch
            with torch.no_grad():
                model.eval()
                eval_inputs, eval_outputs = shuffle_data(eval_inputs, eval_outputs)
                n_batches = (eval_inputs.size(0) // args.batch_size)+1
                sum_loss = 0

                for i_batch in range(n_batches):
                    batch_inputs = eval_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    batch_outputs = eval_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    if batch_inputs.size(0) == 0:
                        continue
                    predictions = model(batch_inputs).view(-1)
                    sum_loss += eval_f(predictions, batch_outputs).item()

                mean_loss = sum_loss / eval_inputs.size(0)
                cv_losses[i_fold, i_epoch, 0] = torch.mean(train_losses)
                cv_losses[i_fold, i_epoch, 1] = mean_loss
                print('Fold %d, Epoch %d Mean Train / Eval Loss and R^2 Value: %.3f / %.3f / %.3f ' % (i_fold+1, i_epoch+1, cv_losses[i_fold, i_epoch, 0], cv_losses[i_fold, i_epoch, 1], 1 - mean_loss / data_error), end='\r')
        fold_models[i_fold] = model
        print('') # to keep only the final epoch losses from each fold


    final_mean_eval_loss = torch.mean(cv_losses[:, -1, 1])
    final_mean_train_loss = torch.mean(cv_losses[:, -1, 0])
    print(('Mean Train / Eval Loss Across Folds at %d Epochs: %.3f / %.3f' % (args.epochs, final_mean_train_loss, final_mean_eval_loss))+' '*10)

    ''' Ensembling '''
    with torch.no_grad():
        inputs, outputs = shuffle_data(inputs, outputs)
        n_batches = (inputs.size(0) // args.batch_size)+1
        sum_loss = [0]*(args.partitions+1)

        for i_batch in range(n_batches):
            batch_inputs = inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            batch_outputs = outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
            if batch_inputs.size(0) == 0:
                continue
            predictions = torch.cat([model(batch_inputs).view(-1, 1) for model in fold_models], dim=1)
            sum_loss[:-1] = [sum_loss[i] + eval_f(predictions[:,i], batch_outputs).item() for i in range(args.partitions)]
            predictions = torch.mean(predictions, dim=1).view(-1)
            sum_loss[-1] += eval_f(predictions, batch_outputs).item()

        mean_loss = [sum_loss[i] / inputs.size(0) for i in range(args.partitions+1)]
        print(('Loss Of Ensemble Over All Folds at %d Epochs: %s' % (args.epochs, str(mean_loss)))+' '*10)



    ''' Training on All Data '''
    ad_losses = torch.zeros(args.partitions, 1, 2)

    eval_inputs = test_inputs
    eval_outputs = test_outputs
    train_inputs = inputs
    train_outputs = outputs
    data_mean = torch.mean(eval_outputs).detach()
    data_error = eval_f(eval_outputs.detach(), torch.ones(eval_outputs.size(0))*data_mean).detach() / eval_outputs.size(0)

    train_losses = torch.zeros(args.partitions, n_batches)

    if args.loss_sampling:
        logits = torch.zeros(inputs.size(0))
    i_epoch = 0
    try:
        while True:
            i_epoch += 1

            # Train Epoch
            [model.train() for model in fold_models]
            n_batches = (train_inputs.size(0) // args.batch_size)+1
            train_inputs, train_outputs = shuffle_data(train_inputs, train_outputs)
            for i_batch in range(n_batches):
                if args.loss_sampling and i_epoch > 0:
                    batch_indices = torch.multinomial(logits, args.batch_size, replacement=True)
                else:
                    batch_indices = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)

                batch_inputs = train_inputs[batch_indices].cuda()
                batch_outputs = train_outputs[batch_indices].cuda()
                if batch_inputs.size(0) == 0:
                    continue
                for i_model in range(args.partitions):
                    batch_losses = step(fold_models[i_model], batch_inputs, batch_outputs, loss_f, fold_opts[i_model])
                    train_losses[i_model, i_batch] = torch.mean(batch_losses).detach()

                if args.loss_sampling:
                    with torch.no_grad():
                        logits[batch_indices] = batch_losses.cpu()

            # Eval Epoch
            with torch.no_grad():
                [model.eval() for model in fold_models]
                eval_inputs, eval_outputs = shuffle_data(eval_inputs, eval_outputs)
                n_batches = (eval_inputs.size(0) // args.batch_size)+1
                sum_loss = [0]*args.partitions

                for i_batch in range(n_batches):
                    batch_inputs = eval_inputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    batch_outputs = eval_outputs[i_batch*args.batch_size:(i_batch+1)*args.batch_size].cuda()
                    if batch_inputs.size(0) == 0:
                        continue
                    for i_model in range(args.partitions):
                        predictions = fold_models[i_model](batch_inputs).view(-1)
                        sum_loss[i_model] += eval_f(predictions, batch_outputs).item()

                mean_loss = [sum_loss[i] / eval_inputs.size(0) for i in range(args.partitions)]
                ad_losses = torch.cat([ad_losses, torch.zeros(args.partitions, 1, 2).to(ad_losses)], dim=1)
                print(ad_losses[i_fold, -1, 0].shape)
                print(torch.mean(train_losses, dim=1).shape)
                ad_losses[:, -1, 0] = torch.mean(train_losses, dim=1)
                ad_losses[:, -1, 1] = torch.FloatTensor(mean_loss).to(ad_losses)
                print('All Data Epoch %d Mean Train / Test Loss and R^2 Value: %s / %s / %s ' % (i_epoch, str(ad_losses[:, i_epoch, 0].cpu().numpy()), str(ad_losses[:, i_epoch, 1].cpu().numpy()), str([1 - mean_loss[i] / data_error for i in range(args.partitions)])), end='\r')
    except Exception:
        traceback.print_exc()
    finally:
        return EnsembleModel(fold_models), cv_losses.cpu(), ad_losses[1:].cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a prediction over the eval set.')
    parser.add_argument('model_path', type=str, help='Relative path at which to save the model.')
    parser.add_argument('-partitions', type=int, default=3, help='Number of partitions for cross-fold validation.')
    parser.add_argument('-batch_size', type=int, default=256, help='Number of samples per batch.')
    parser.add_argument('-batch_size_annealing', type=float, default=1.0, help='Per-batch multiplier on batch size.')
    parser.add_argument('-epochs', type=int, default=25, help='Number of epochs to train for.')
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-loss_sampling', default=False, action='store_true', dest='loss_sampling', help='Flag to use loss-based sampling in place of traditional batching.')

    args = parser.parse_args()

    try:
        model = torch.load(args.model_path + '/model.ptm').cuda()
    except:
        model = build_model(args)
    inputs = torch.load('train_in.pt')
    outputs = torch.load('train_out.pt')

    os.makedirs(args.model_path, exist_ok=True)

    model, cvloss, adloss = train(model, inputs, outputs, args)

    save_losses(cvloss, args.model_path+'/cross_validation_losses')
    save_losses(adloss, args.model_path+'/all_data_losses')

    torch.save(model.cpu(), args.model_path + '/model.ptm')
