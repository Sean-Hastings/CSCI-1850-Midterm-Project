import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, norm_mean=0.0, norm_std=1.0):
        super(Model, self).__init__()
        if isinstance(norm_mean, float):
            self.register_buffer('norm_mean', torch.FloatTensor([norm_mean]))
            self.register_buffer('norm_std', torch.FloatTensor([norm_std]))
        else:
            self.register_buffer('norm_mean', norm_mean)
            self.register_buffer('norm_std', norm_std)

    def norm(self, values):
        return (values - self.norm_mean) / self.norm_std


class ConvBlock(nn.Module):
    def __init__(self, *args, batch_norm=False, dropout=0, deconv=False, activation='leaky_relu', **kwargs):
        super(ConvBlock, self).__init__()
        self.activation = activation
        if deconv:
            self.conv = nn.ConvTranspose1d(*args, **kwargs)
        else:
            self.conv = nn.Conv1d(*args, **kwargs)
        self.dropout = nn.Dropout2d(dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(args[1])
        else:
            self.batch_norm = False

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        if self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'sigmoid':
            x = F.sigmoid(x)
        elif self.activation == 'softmax':
            x = torch.softmax(x, dim=1)
        return x


class ConvolutionalModel(Model):
    def __init__(self, norm_mean=0.0, norm_std=1.0, n_outs=1, in_size=15, activation=None):
        super(ConvolutionalModel, self).__init__(norm_mean, norm_std)
        self.conv = nn.ModuleList([
            ConvBlock(in_size, 256, 3, batch_norm=True, dropout=.2, stride=1),
            ConvBlock(256, 256, 3, batch_norm=True, dropout=.2, stride=2),
            ConvBlock(256, 128, 3, batch_norm=True, dropout=.2, stride=1),
            ConvBlock(128, 128, 3, batch_norm=True, dropout=.2, stride=2),
            ConvBlock(128, 256, 3, batch_norm=True, dropout=.2, stride=1),
            ConvBlock(256, 256, 3, batch_norm=True, dropout=.2, stride=2),
            ConvBlock(256, 512, 3, batch_norm=True, dropout=.2, stride=1),
            ConvBlock(512, 512, 3, batch_norm=True, dropout=.2, stride=2),
            ConvBlock(512, 1024, 3, batch_norm=True, dropout=.5),
            ConvBlock(1024, 1024, 1, batch_norm=True, dropout=.5),
            ConvBlock(1024, n_outs, 1, activation=activation)
        ])

    def forward(self, inputs, show=False, eval=False, use_classifier=False):
        x = [self.norm(inputs)]
        for conv in self.conv:
            x += [conv(x[-1])]
        x[-1] = x[-1].view(inputs.size(0), -1)
        if show:
            return x
        else:
            return x[-1]


class MixModel(Model):
    def __init__(self, sequence_channels=256, rnn_layers=1, embed_size=16):
        super(MixModel, self).__init__()
        self.conv = ConvolutionalModel(in_size=15+sequence_channels*2)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=sequence_channels, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.embed = nn.Embedding(num_embeddings=5, embedding_dim=embed_size)

    def forward(self, histones, sequences, eval=False):
        x = self.embed(sequences)
        x = self.rnn(x)
        x = x[0]
        x = x[:,50::100]
        x = x.permute(0, 2, 1)
        x = self.conv(torch.cat([histones, x], dim=1))
        return x


class SeqEncoder(Model):
    def __init__(self, norm_mean=0.0, norm_std=1.0, n_outs=1, embed_size=8):
        super(SeqEncoder, self).__init__(norm_mean, norm_std)
        self.embed = nn.Embedding(num_embeddings=5, embedding_dim=embed_size)
        self.feature_extractor = ConvBlock(embed_size, 64, 3, batch_norm=True, dropout=.2, stride=1)
        self.conv1 = ConvBlock(64, 128, 5, batch_norm=True, dropout=.2, stride=3)
        self.conv2 = ConvBlock(128, 256, 7, batch_norm=True, dropout=.2, stride=3)
        self.conv3 = ConvBlock(256, 512, 9, batch_norm=True, dropout=.2, stride=3)
        self.conv4 = ConvBlock(512, n_outs, 11, batch_norm=True, dropout=0, stride=3, activation='sigmoid')
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, inputs, n_layers=30):
        if n_layers > len(self.conv_layers):
            n_layers = len(self.conv_layers)
        x = self.feature_extractor(self.embed(inputs).permute(0, 2, 1))
        for i in range(n_layers):
            x = self.conv_layers[i](x)
        return x


class SeqDecoder(Model):
    def __init__(self, norm_mean=0.0, norm_std=1.0, in_size=15, embed_size=8):
        super(SeqDecoder, self).__init__(norm_mean, norm_std)
        self.conv1 = ConvBlock(in_size, 512, 11, batch_norm=True, dropout=.2, stride=3, deconv=True)
        self.conv2 = ConvBlock(512, 256, 9, batch_norm=True, dropout=.2, stride=3, deconv=True, output_padding=1)
        self.conv3 = ConvBlock(256, 128, 7, batch_norm=True, dropout=.2, stride=3, deconv=True, output_padding=1)
        self.conv4 = ConvBlock(128, 64, 5, batch_norm=True, dropout=.2, stride=3, deconv=True, output_padding=2)
        self.feature_detractor = ConvBlock(64, embed_size, 3, batch_norm=True, dropout=0, stride=1, deconv=True)
        self.embed = ConvBlock(embed_size, 4, 1, batch_norm=True, dropout=0, stride=1, deconv=False, activation='softmax')
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, inputs, n_layers=30):
        if n_layers > len(self.conv_layers):
            n_layers = len(self.conv_layers)
        x = inputs
        for i in range(len(self.conv_layers)-n_layers, len(self.conv_layers)):
            x = self.conv_layers[i](x)
        x = self.embed(self.feature_detractor(x))
        return x


class SeqModel(nn.Module):
    def __init__(self, encode_size=256, embed_size=8):
        super(SeqModel, self).__init__()
        self.encoder = SeqEncoder(n_outs=encode_size, embed_size=embed_size)
        self.decoder = SeqDecoder(in_size=encode_size, embed_size=embed_size)

    def forward(self, inputs, n_layers=30):
        encoding = self.encoder(inputs, n_layers)
        reconstruction = self.decoder(encoding, n_layers)
        return reconstruction, encoding


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.classifier = None
        if isinstance(models[-1], ClassificationModel):
            self.models = nn.ModuleList(models[:-1])
            self.classifier = models[-1]
        else:
            self.models = nn.ModuleList(models)

    def forward(self, inputs, use_classifier=True):
        predictions = [model(inputs, eval=True) for model in self.models]
        shape = predictions[0].shape
        predictions = torch.cat([pred.view(shape[0], -1, 1) for pred in predictions], dim=2)
        if self.classifier is not None and use_classifier:
            weights = self.classifier(inputs).unsqueeze(1)
            mean_prediction = torch.sum(predictions * weights, dim=2).view(shape)
        else:
            mean_prediction = torch.mean(predictions, dim=2).view(shape)
        return mean_prediction


class ClassificationModel(Model):
    def __init__(self, norm_mean=0.0, norm_std=1.0, n_classes=48, in_size=15):
        super(ClassificationModel, self).__init__(norm_mean, norm_std)
        self.conv_stack = ConvolutionalModel(norm_mean, norm_std, n_classes, in_size, activation='softmax')

    def forward(self, inputs, show=False, eval=False):
        activations = self.conv_stack(inputs, show)
        if show:
            activations[-1] = activations[-1]
        else:
            activations = activations
        return activations
