
DIM_MODEL = 128
NUM_HEAD = 4
DROPOUT = 0.5

BATCH_SIZE = 16
NUM_EPOCHS = 100
TEACHER_FORCING_RATE = 0.0
LEARNING_RATE = 0.001
PATIENCE_THRESHOLD = 4

SMILES_PATH = '../data/ADAGRASIB_SMILES.txt'
COORDINATE_PATH = '../data/ADAGRASIB_COOR.sdf'






from torch.utils.data import Dataset, DataLoader 
from utils import *
from model import * 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


class Coor2SmiDataset(Dataset) :
    def __init__(self, coor_list, smint_list) :
        self.smint_list = torch.tensor(smint_list, dtype = torch.long, device=device)
        self.coor_list = [torch.tensor(coor, device=device) for coor in coor_list]

    def __len__(self) :
        return len(self.smint_list)
    
    def __getitem__(self, idx) :
        return self.coor_list[idx], self.smint_list[idx]
    

smi_list = get_smi(SMILES_PATH)
longest_smi = get_longest(smi_list)
smi_dic = get_dic(smi_list)
inv_smi_dic = {value:key for key, value in smi_dic.items()}
smint_list = [smi2int(smi, smi_dic, longest_smi) for smi in smi_list]


coor_list = get_coor(COORDINATE_PATH)
longest_coor = get_longest(coor_list)
np_coor_list = pad_coor(normalize_coor(coor_list), longest_coor)


train_smint, val_smint, test_smint = split_data(smint_list)
train_coor, val_coor, test_coor = split_data(np_coor_list)


train_set = Coor2SmiDataset(train_coor, train_smint)
val_set = Coor2SmiDataset(val_coor, val_smint)
test_set = Coor2SmiDataset(test_coor, test_smint)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


encoder = Encoder(DIM_MODEL, NUM_HEAD, DROPOUT).to(device)
decoder = Decoder(DIM_MODEL, NUM_HEAD, len(smi_dic),longest_smi, DROPOUT).to(device)

weight = torch.ones(len(smi_dic)).to(device)
weight[2] = torch.tensor(0.0001)

train(train_loader, val_loader, test_loader,
      encoder, decoder,
      weight=weight,
      num_epoch=NUM_EPOCHS,
      learning_rate=LEARNING_RATE,
      patience_threshold=PATIENCE_THRESHOLD,
      tf_rate = TEACHER_FORCING_RATE)


torch.save(encoder.state_dict(),f'./trained-model/Encoder-D{DIM_MODEL}-H{NUM_HEAD}-DROPOUT{DROPOUT}-WEIGHTADJUST.pth')
torch.save(decoder.state_dict(),f'./trained-model/Decoder-D{DIM_MODEL}-H{NUM_HEAD}-DROPOUT{DROPOUT}-WEIGHTADJUST.pth')