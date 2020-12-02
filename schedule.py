def schedule(epoch):
    
    if epoch < 3:
         new_lr = .001
    elif epoch < 5:
         new_lr = .0006
    elif epoch < 7:
         new_lr = .0003
    elif epoch < 9:
         new_lr = .0001
    elif epoch < 12:
         new_lr = .00005
    else:
         new_lr = .00001
    
    print("\nLR at epoch {} = {}  \n".format(epoch,new_lr))
    return new_lr