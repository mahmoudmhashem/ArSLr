import torch
from torch import nn
class MyModel(nn.Module):
    def __init__(self, reset2traincontext=True, device = 'cuda'):
        super(MyModel, self).__init__()

        self.classes = ['baby', 'eat', 'father', 'finish', 'good', 'happy', 'hear', 
                        'house', 'important', 'love', 'mall', 'me', 'mosque', 'mother', 
                        'normal', 'sad', 'stop', 'thanks', 'thinking', 'worry']
        
        self.class_to_idx = { "baby": 0, "eat": 1, "father": 2, "finish": 3, 
                        "good": 4, "happy": 5, "hear": 6, "house": 7, "important": 8, 
                        "love": 9, "mall": 10, "me": 11, "mosque": 12, "mother": 13, 
                        "normal": 14, "sad": 15, "stop": 16, "thanks": 17, "thinking": 18, "worry": 19 }

        self.device = device      
        self.n_features = [134, 64, 32, 20]

        self.headen_state1 = torch.zeros(1, 1, self.n_features[1], device=self.device)
        self.headen_state2 = torch.zeros(1, 1, self.n_features[2], device=self.device)
        self.headen_state3 = torch.zeros(1, 1, self.n_features[3], device=self.device)
        

        self.gru1 =  nn.GRU(self.n_features[0], self.n_features[1], batch_first=True, device=self.device)
        self.linear1 =  nn.Linear(self.n_features[1], self.n_features[1], device=self.device)
        self.relu1 =  nn.ReLU()
        self.gru2 =  nn.GRU(self.n_features[1], self.n_features[2], batch_first=True, device=self.device)
        self.linear2 =  nn.Linear(self.n_features[2], self.n_features[2], device=self.device)
        self.relu2 =  nn.ReLU()
        self.gru3 =  nn.GRU(self.n_features[2], self.n_features[3], batch_first=True, device=self.device)
        self.linear3 =  nn.Linear(self.n_features[3], self.n_features[3], device=self.device)
        self.eval()

        if reset2traincontext:
            self.reset2traincontext()

    def forward(self, x):
        x, self.headen_state1 = self.gru1(x, self.headen_state1)
        x = self.linear1(x)
        x = self.relu1(x)

        x, self.headen_state2 = self.gru2(x, self.headen_state2)
        x = self.linear2(x)
        x = self.relu2(x)

        x, self.headen_state3 = self.gru3(x, self.headen_state3)
        x = self.linear3(x)
        return x
    
    @torch.jit.export
    def predict(self, x):
        x = x.to(self.device)
        logits = self.forward(x)

        b, s, f = logits.shape
        logits = logits.view(b*s, f)

        predictions = torch.softmax(logits, dim=1).argmax(dim=1)
        labels = [self.classes[idx] for idx in predictions]
        return {'labels':labels, 'indices':predictions}

    @torch.jit.export
    def clearcontext(self):
        self.headen_state1 = torch.zeros(1, 1, self.n_features[1], device=self.device)
        self.headen_state2 = torch.zeros(1, 1, self.n_features[2], device=self.device)
        self.headen_state3 = torch.zeros(1, 1, self.n_features[3], device=self.device)
    
    @torch.jit.export
    def on_batch(self):
        self.headen_state1 = self.headen_state1.detach()
        self.headen_state2 = self.headen_state2.detach()
        self.headen_state3 = self.headen_state3.detach()
    
    @torch.jit.export
    def reset2traincontext(self):
        self.headen_state1 = torch.tensor([[[ 0.4482, -0.4596, -0.8315,  0.5823, -0.5038, -0.6898,  0.3295,
                                            -0.9982, -0.9350, -0.3410, -0.6797,  0.9997, -0.1166, -0.9870,
                                            0.6391, -0.6688,  0.3455,  0.2062,  0.5480, -0.6831, -0.2516,
                                            -0.8817,  0.5530,  0.1282,  0.7358,  0.9835,  0.5293,  1.0000,
                                            -0.5893,  0.2882,  0.9891, -0.6322, -0.4782, -0.9961, -0.6503,
                                            0.3842, -0.9100,  0.0490, -0.0761, -0.4697,  0.9987,  0.9590,
                                            -0.7767, -0.9236,  0.9872, -0.3746,  0.5919,  0.4405,  0.9125,
                                            -0.9421,  0.5804, -0.1957, -0.0049,  0.6423,  0.6667,  0.5910,
                                            0.8491, -0.0031, -1.0000, -0.0227,  0.8973, -0.7942,  0.6784,
                                            -0.2965]]], device=self.device)

        self.headen_state2 = torch.tensor([[[-0.9271,  0.9930, -0.9673,  0.9996,  0.4764, -0.9968,  0.8065,
                                            0.8920, -1.0000,  0.8556,  0.9838, -0.6858, -0.6497,  0.4286,
                                            1.0000,  0.8423, -0.9785, -0.9917,  1.0000, -0.2412, -0.5915,
                                            -0.8412, -0.8129,  0.9982,  0.1080, -0.6362, -1.0000,  0.9488,
                                            0.9753,  1.0000,  1.0000,  0.9888]]], device=self.device)
        
        self.headen_state3 = torch.tensor([[[-0.9939, -0.1616,  0.9998, -0.6852,  0.8527, -0.9799,  0.9565,
                                            0.9207,  0.7528, -0.5844,  0.8351,  0.9999, -0.8931,  0.9916,
                                            0.9398, -1.0000, -0.8999,  0.2363, -0.9698,  0.9998]]], 
                                            device=self.device)