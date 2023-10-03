import torch
def MBE(m1, bit_redundancy = 3):
        N,M = m1.shape
        new_m = torch.zeros((N,int(M//8)*3))
        for iii in range(N):
            x = m1[iii].chunk(32//8)
            for iiii in range(32//8):
                new_m[iii][iiii*bit_redundancy:(iiii+1)*bit_redundancy] = x[iiii][:bit_redundancy]
        return new_m

a = torch.randint(0,255,(2,32))
print(a)
aa = MBE(a)
print(aa)