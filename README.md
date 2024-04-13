# This is a midterm project for a Deep Learning class at NYU Tandon. 

The codes are built upon this repository: https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min and https://github.com/kuangliu/pytorch-cifar
The datasets are from this link: https://www.cs.toronto.edu/~kriz/cifar.html

Please install the necessary libraries with the requirements.txt file provided in the repository.

The final_kaggle_train.py file can be run to create a checkpoint that we submitted to the competition. In addition, the checkpoint can be found at this link: https://drive.google.com/file/d/1wbhihNJfBIh9eSSvM8fm1bvOatX-UTfm/view?usp=sharing.

The main.py file is what we used to perform various experiments for hyperparameter settings. It should not be run by itself (as it won't produce any meaningful checkpoints). The reamining scripts are all utility scripts.

The best result (84.7% on Kaggle test) we have is using ResNet5M model with these hyperparameters: 

    total_train_size = 60000 (no validation set)
    train_batch_size = 400
    epochs = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 
                            lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

In addition, we retrained the above model with a validation set to produce proper visualizations in the report. It achied 79.7% on the Kaggle test and uses the following hyperparameters:

    total_train_size = 45000 
    total_valid_size = 5000
    train_batch_size = 400
    epochs = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 
                            lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


