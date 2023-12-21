import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import rdkit
from rdkit import Chem
import math 
import time 
import numpy as np 
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_all_atom(smi_path) :
    smi_dic = {}
    i = 0

    with open(smi_path, 'r') as file :
        for smi in file :
            smi = rdkit.Chem.MolFromSmiles(smi)
            for atom in smi.GetAtoms() :
                atom = atom.GetSymbol()
                if atom not in smi_dic : 
                    smi_dic[atom] = i 
                    i += 1
    return smi_dic

def replace_atom(input, mode = 'train') :
    if mode == 'train' :
        smi_list = [smi.replace('Cl', 'X')
                    .replace('Br', 'Y')
                    .replace('Na', 'Z')
                    .replace('Ba', 'T') for smi in input]
        return smi_list
        
    if mode == 'eval' :
        smi = input.replace('X', 'Cl').replace('Y', 'Br').replace('Z', 'Na').replace('T', 'Ba').replace('x','')
        return smi

def get_longest(input_list) :
    longest = 0
    for i in input_list :
        if len(i) > longest :
            longest = len(i)
    return longest

def get_coor(coor_path) :
    coor_list = []
    supplier = rdkit.Chem.SDMolSupplier(coor_path)
    for mol in supplier:
        coor = []
        if mol is not None:
            conformer = mol.GetConformer()
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                x, y, z = conformer.GetAtomPosition(atom_idx)
                coor_atom = list((x,y,z))
                coor.append(coor_atom)
        coor_list.append(coor)

    # Replace invalid idx
    for i, coor in enumerate(coor_list):
        
        if len(coor) == 0 :
            if i == 0 :
                coor_list = coor_list[1:]
            coor_list[i] = coor_list[i-1]
    return coor_list

def get_smi(smi_path) :
    smi_list = []
    i = 0
    with open(smi_path, 'r') as file :
        for smi in file :
            if rdkit.Chem.MolFromSmiles(smi) is None :
                if len(smi_list) == 0 :
                    continue 
                smi_list.append(smi_list[i-1])
                i += 1 
                continue 
            smi_list.append(smi)
            i += 1
    
    smi_list = [smi[:-1] for smi in smi_list]
    smi_list = [smi + 'E' for smi in smi_list]
    smi_list = replace_atom(smi_list)
    return smi_list

def get_dic(smi_list) :
    smi_dic = {'x': 0,
               'E': 1}
    i = len(smi_dic)

    for smi in smi_list : 
        for atom in smi :
            if atom not in smi_dic : 
                smi_dic[atom] = i
                i += 1 
    return smi_dic 

def count_atoms(smi):
    smi = replace_atom(smi, mode = 'eval')
    mol = rdkit.Chem.MolFromSmiles(smi)
    if mol is not None:
        num_atoms = mol.GetNumAtoms()
        return num_atoms
    else:
        print("Error: Unable to parse SMILES string.")
        return None
    
def smi2int(smi, smi_dic, longest_smi, mode = 'data') :
    if mode == 'eval' :
        # smi += 'E'
        smi = smi + 'E' if smi[-1] != 'E' else smi
        smi = list(smi)
        smint = [smi_dic[atom] for atom in smi]
        smint = smint + [0] * (longest_smi - len(smint))
        smint = torch.tensor(smint, dtype=torch.long, device = device)
        smint = smint.unsqueeze(0)
        return smint
    smi = list(smi)
    smint = [smi_dic[atom] for atom in smi]
    smint = smint + [0] * (longest_smi - len(smint))
    return smint 

def int2smi(smint, inv_smi_dic) :
    smint = smint.cpu().numpy()
    
    smi = [inv_smi_dic[i] for i in smint] 
    smi = ''.join(smi)
    smi = replace_atom(smi, mode = 'eval')
    return smi


def normalize_coor(coor_list) :
    n_coor_list = []

    for mol_coor in coor_list :
        n_mol_coor = []

        x_origin, y_origin, z_origin = mol_coor[0]

        for atom_coor in mol_coor :
            n_atom_coor = [round(atom_coor[0] - x_origin, 2), 
                        round(atom_coor[1] - y_origin, 2), 
                        round(atom_coor[2] - z_origin, 2)]
            n_mol_coor.append(n_atom_coor)
        n_coor_list.append(n_mol_coor)
    return n_coor_list

def pad_coor(coor_list, longest_coor) :
    p_coor_list = []

    for i in coor_list :
        if len(i) < longest_coor :
            zeros = [[0,0,0]] * (longest_coor - len(i))
            zeros = torch.tensor(zeros)
            i = torch.tensor(i)
            i = torch.cat((i, zeros), dim = 0)
            p_coor_list.append(i)
        else :
            p_coor_list.append(i)
    return p_coor_list

def split_data(input, ratio = [0.9,0.05,0.05]) :
    assert sum(ratio) == 1, "Ratio does not add up to 1"  
    stop1 = int(len(input) * ratio[0])
    stop2 = int(len(input) * (ratio[0] + ratio[1])) 

    train = input[:stop1]
    val = input[stop1:stop2]
    test = input[stop2:]

    return train, val, test
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluate(input, encoder, decoder, smi_dic, longest_smi) :
    encoder.eval(), decoder.eval()
    if type(input) == str :
        input = smi2int(input, smi_dic, longest_smi, mode = 'eval')
    
    with torch.no_grad() :
        e_all, e_last, self_attn = encoder(input) 
        prediction, cross_attn = decoder(e_all, e_last)

        prediction, self_attn, cross_attn = prediction.squeeze(0).cpu().numpy(), self_attn.squeeze(0), cross_attn.squeeze(0)
    return prediction, self_attn, cross_attn


def plot_attn(matrix, smi, mode, output_path = "", output_name = "", output_type = "show") :

    if mode == "cross" :
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix, cmap = "viridis")
        fig.colorbar(cax)
        ax.set_xticklabels([''] + list(smi))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


    if mode == "self" :
        num_head = matrix.shape[0]
        
        fig, ax = plt.subplots(1, num_head, figsize=(num_head*10,30))
        for i, head in enumerate(matrix) :
            cax = ax[i].matshow(head, cmap='viridis')
            # fig.colorbar(cax)
            ax[i].set_xticklabels([''] + list(smi), fontsize="xx-large")
            ax[i].set_yticklabels([''] + list(smi), fontsize='xx-large')
            
            ax[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax[i].yaxis.set_major_locator(ticker.MultipleLocator(1))
    if output_type == "show" :
        plt.show()
    if output_type == "save" :
        plt.savefig(f"{output_path}/{output_name}")
    plt.close()
    
def visualize(smi,
              encoder, decoder,
              smi_dic, longest_smi,
              output_path = "",
              output_name="test",
              output_type="show") :
    
    
    prediction, self_attn, cross_attn = evaluate(smi, encoder, decoder, smi_dic, longest_smi)

    self_attn = self_attn.cpu().numpy()
    cross_attn = cross_attn.cpu().numpy()

    smi = smi[:-1] if smi[-1] == 'E' else smi

    # smi = replace_atom(smi, mode='eval')

    coor_len = count_atoms(smi)
    smi_len = len(smi) 

    plot_attn(self_attn[:, :smi_len, :smi_len],smi=smi, mode='self', output_path=output_path, output_name=f"{output_name}-SELF", output_type=output_type)
    plot_attn(cross_attn[:coor_len, :smi_len],smi=smi, mode='cross', output_path=output_path, output_name=f"{output_name}-CROSS", output_type=output_type)


r1 = random.randint(0, 4000)
r2 = random.randint(0, 4000)
r3 = random.randint(0, 4000)

def train_epoch(train_loader, val_loader, test_loader,
                encoder, decoder,
                encoder_optimizer, decoder_optimizer,
                criterion, tf):

    epoch_train_loss = 0
    epoch_val_loss = 0

    for input, target in train_loader:

        encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad()
        
        e_all, e_last, self_attn = encoder(input)

        # Teacher Forcing
        if tf :
          prediction, cross_attn = decoder(e_all, e_last, target)
        else :
          prediction, cross_attn = decoder(e_all, e_last)


        loss = criterion(prediction, target)
        loss.backward()

        encoder_optimizer.step(), decoder_optimizer.step()
        
        epoch_train_loss += loss.item()


    encoder.eval(), decoder.eval()
    


    with torch.no_grad() :
      for input, target in val_loader :
        e_all, e_last, self_attn = encoder(input)
        prediction, cross_attn = decoder(e_all, e_last)

        test_loss = criterion(prediction, target)
        epoch_val_loss += test_loss.item()

    return epoch_train_loss / len(train_loader), epoch_val_loss / len(test_loader)

def train(train_loader, val_loader, test_loader,
          encoder, decoder, 
          smi_list, smi_dic, longest_smi, output_path,
          patience_threshold = 4,num_epoch=50, learning_rate=0.001, tf_rate = 0):
    start = time.time()
    
    best_val = float('inf')
    patience = 0

    train_plot, val_plot = [], []

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.L1Loss()

    tf = True

    for epoch in range(1, num_epoch + 1):
      if epoch > (tf_rate * num_epoch) :
        tf = False
      encoder.train()
      decoder.train()

      train_loss, val_loss = train_epoch(train_loader,val_loader, test_loader,
                                         encoder, decoder,
                                         encoder_optimizer, decoder_optimizer,
                                         criterion, tf)

      print('%s (%d %d%%) /// Train Loss: %.4f - Validation Loss: %.4f' % (timeSince(start, epoch / num_epoch),
                                      epoch, epoch / num_epoch * 100, train_loss, val_loss))

      if val_loss < best_val :
        best_val = val_loss
        patience = 0
      else :
        patience += 1 
      
      if patience > patience_threshold : 
        print("EARLY STOPPING !!!")
        plt.plot(x, train_plot, color = 'blue', label = 'Train Loss')
        plt.plot(x, val_plot, color = 'red', label = 'Validation Loss')
        plt.title("Final Plot Before Stop")
        plt.legend()
        plt.show(block=False)
        plt.savefig(f'./loss-image/EPOCH {epoch}')
        break


      

      train_plot.append(train_loss), val_plot.append(val_loss)
      x = np.linspace(0, num_epoch, epoch)
      if epoch == 1 : 
        continue
      if epoch % 5 == 0 :
        plt.clf()
        plt.plot(x, train_plot, color = 'blue', label = 'Train Loss')
        plt.plot(x, val_plot, color = 'red', label = 'Validation Loss')
        plt.title(f'Epoch {epoch}')
        plt.legend()
        plt.show(block=False)
        plt.savefig(f'./loss-image/EPOCH {epoch}')
        plt.pause(5)
        plt.close()
        visualize(smi_list[r1], encoder, decoder, smi_dic, longest_smi, output_path=output_path, output_name=f"R1-E{epoch}",output_type='save')
        visualize(smi_list[r2], encoder, decoder, smi_dic, longest_smi, output_path=output_path, output_name=f"R2-E{epoch}", output_type='save')
        visualize(smi_list[r3], encoder, decoder, smi_dic, longest_smi, output_path=output_path, output_name=f"R3-E{epoch}", output_type='save')
          
