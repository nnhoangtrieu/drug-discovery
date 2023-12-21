import selfies

test = 'COc1ccc(cc1)/C=C/C(=O)N1CCCCC1'
out = selfies.encoder(test)
print(type(out))