import selfies
import re 
smi_dic = {}
i = 0


# Define a regular expression pattern to match substrings enclosed in brackets
pattern = re.compile(r'\[([^]]*)\]')



with open('./data/ADAGRASIB_SMILES.txt', 'r')  as file :
    for smi in file :
        sel = selfies.encoder(smi[:-1])
        matches = pattern.findall(sel)
        for atom in matches :
            if atom not in smi_dic :
                smi_dic[atom] = i 
                i += 1


    


print(smi_dic)


# import re

# input_string = '[C][N][N][=C][C][=Branch1][Ring2][=C][Ring1][Branch1][B][O][C][Branch1][O][C][Branch1][Ring2][O][Ring1][Branch1][Branch1][C][C][C][Branch1][C][C][C]'

# # Define a regular expression pattern to match substrings enclosed in brackets
# pattern = re.compile(r'\[([^]]*)\]')

# # Use findall to extract all matches
# matches = pattern.findall(input_string)

# print(matches)
