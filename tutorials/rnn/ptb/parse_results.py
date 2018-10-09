import sys
import decimal


def f2s(f):
    ctx = decimal.Context()
    ctx.prec = 20
    d1 = ctx.create_decimal(repr(f))
    st = format(d1, 'f')
    if '.' not in st:
        st += '.0'
    return st


def parse_results(filename):
    lines = open(filename).readlines()
    lines = [l.strip() for l in lines if l.strip() != ""]

    comm_str = lines[0]
    lines = lines[1:]

    min_val_per = 100000
    min_val_per_split = ''
    min_te_per = -1
    min_te_per_split = ''
    epoch = -1
    val_updated = False
    for l in lines:
        if 'Valid Perplexity:' in l:
            el = l.split()
            per = float(el[4])

            if per < min_val_per:
                min_val_per = per
                min_val_per_split = el[5]
                epoch = el[1]
                val_updated = True

        if 'Test Perplexity:' in l and val_updated:
            el = l.split()
            per = float(el[4])
            min_te_per = per
            min_te_per_split = el[5]
            val_updated = False

    print(comm_str)
    print('Epoch: '+epoch)
    print("val perplexicity: "+str(min_val_per)+" "+str(min_val_per_split))
    print("tes perplexicity: "+str(min_te_per)+"    tes prec@5: "+str(min_te_per_split))
    print("")
    print("")


if __name__ == '__main__':
    file_prefix = sys.argv[1]

    MOD='custom'

    HIDDEN_SIZE_LIST=[50]
    LR_LIST=[1.0]
    LR_DECAY_LIST=[0.5, 0.6, 0.7, 0.8]
    EXP_LIST=[1.0, 1.3, 1.5]
    BATCH_SIZE_LIST=[20]
    MME=40
    MAX_EPOCH_LIST=[2, 4, 6]
    LFUNC="logistic"
    NSTEPS=20

    for HS in HIDDEN_SIZE_LIST:
        for EX in EXP_LIST:
            for LRATE in LR_LIST:
                for LRD in LR_DECAY_LIST:
                    for BS in BATCH_SIZE_LIST:
                        for ME in MAX_EPOCH_LIST:
                            OUT=file_prefix+'_HS'+str(HS)+'_LR'+str(LRATE)+'_LRD'+str(LRD)+'_BS'+str(BS)+'_ME'+str(ME)+'_EXP'+str(EX)
                            print('Hidden Size: '+str(HS)+' Exp: '+str(EX)+' LR: '+str(LRATE)+' LR Decay: '+str(LRD)+' Batch Size: '+str(BS)+' Max epoch: '+str(ME))
                            parse_results(OUT)
                            print('')
            print('')
            print('')
        
        print('')
        print('')
        print('')
        print('')



