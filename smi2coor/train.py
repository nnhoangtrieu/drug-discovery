
DIM_MODEL = 128
NUM_BLOCK = 1
NUM_HEAD = 4
DROPOUT = 0.5
FE = 1

BATCH_SIZE = 16
NUM_EPOCHS = 100
TEACHER_FORCING_RATE = 0.0
LEARNING_RATE = 0.001
PATIENCE_THRESHOLD = 4
ATTENTION_IMAGE_OUTPUT_PATH = 'attention-image'

SMILES_PATH = '../data/ADAGRASIB_SMILES.txt'
COORDINATE_PATH = '../data/ADAGRASIB_COOR.sdf'






from torch.utils.data import Dataset, DataLoader 
from utils import *
from model import * 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


class SMILES2CoorDataset(Dataset) :
    def __init__(self, smi_list, coor_list) :
        self.smi_list = torch.tensor(smi_list, dtype = torch.long, device=device)
        self.coor_list = [torch.tensor(coor, device=device) for coor in coor_list]

    def __len__(self) :
        return len(self.smi_list)
    
    def __getitem__(self, idx) :
        return self.smi_list[idx], self.coor_list[idx]
    

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


train_set = SMILES2CoorDataset(train_smint, train_coor)
val_set = SMILES2CoorDataset(val_smint, val_coor)
test_set = SMILES2CoorDataset(test_smint, test_coor)


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


encoder = Encoder(dim_model=DIM_MODEL,
                  num_block=NUM_BLOCK,
                  num_head=NUM_HEAD,
                  dropout=DROPOUT,
                  fe = FE,
                  len_dic=len(smi_dic)).to(device)

decoder = Decoder(dim_model=DIM_MODEL,
                  num_block=NUM_BLOCK,
                  num_head=NUM_HEAD,
                  dropout=DROPOUT,
                  fe=FE,
                  longest_coor=longest_coor,
                  ).to(device)


train(train_loader,val_loader, test_loader,
      encoder, decoder,
      smi_list, smi_dic, longest_smi, 
      output_path=ATTENTION_IMAGE_OUTPUT_PATH, 
      patience_threshold=PATIENCE_THRESHOLD,
      num_epoch=NUM_EPOCHS,
      learning_rate=LEARNING_RATE,
      tf_rate = TEACHER_FORCING_RATE)


torch.save(encoder.state_dict(),f'./trained-model/Encoder-D{DIM_MODEL}-B{NUM_BLOCK}-H{NUM_HEAD}-FE{FE}-DROPOUT{DROPOUT}.pth')
torch.save(decoder.state_dict(),f'./trained-model/Decoder-D{DIM_MODEL}-B{NUM_BLOCK}-H{NUM_HEAD}-FE{FE}-DROPOUT{DROPOUT}.pth')