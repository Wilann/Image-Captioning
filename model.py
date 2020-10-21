import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):        
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        # Remove <end> 
        captions = captions[:, :-1] 
        
        # Embed caption to fixed size 
        captions = self.embed(captions)  # (batch_size, num_captions, vocab_size) -> (batch_size, num_captions, embed_size) 
        
        # Concatenate image's feature vectors and embedings 
        # features.shape:              (batch_size, embed_size) 
        # features.unsqueeze(1).shape: (batch_size, 1, embed_size)
        # captions.shape:              (batch_size, num_captions, embed_size) 
        # inputs.shape:                (batch_size, num_captions, embed_size)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # Sent features and embeddings through LSTM 
        outputs, _ = self.lstm(inputs)  # (batch_size, num_captions, embed_size) -> (batch_size, num_captions, hidden_size)
        
        # Fully-connected Layer 
        outputs = self.fc(outputs)  # (batch_size, num_captions, hidden_size) -> (batch_size, num_captions, vocab_size)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []   
        output_length = 0
        
        while (output_length != max_len+1):
            
            # Feed embeddings into LSTM 
            output, states = self.lstm(inputs, states)
            
            # Fully-connected layer to output probability distribution of words 
            output = self.fc(output.squeeze(dim=1))
            
            # Get predicted word - the one with the highest probability 
            _, predicted_index = torch.max(output, 1)
            
            outputs.append(predicted_index.cpu().numpy()[0].item())

            # <end> has index 1
            # If <end> is predicted, there are no more words to predict 
            if (predicted_index == 1):
                break
            
            # Prepare to predict next word 
            inputs = self.embed(predicted_index)   
            inputs = inputs.unsqueeze(1)
            
            # Keep track of how long current output is 
            output_length += 1

        return outputs