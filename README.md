# This is a midterm project for a Deel Learning class at NYU Tandon. 

The codes are built upon this repository: https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min
The datasets are from this link: https://www.cs.toronto.edu/~kriz/cifar.html

Here is a short description and objectives of this project: \
*Dataset*: The test data, named cifar_test_nolabels, consists of images from the CIFAR10 dataset, but from a different source and without the correct labels. \
*Task*: You are required to run inference on this dataset using your trained model and submit a CSV file containing your model's predictions. Ensure that each prediction corresponds accurately to the ID of the test image. \
*Submission*: Upon successful submission of your CSV, your model's accuracy will be displayed, and you will be ranked based on this score. \
*Objective*: The key focus is on the generalization capabilities of your model. It's crucial to avoid overfitting to ensure that your model performs well on unseen data. \
*Competition/Submission website*: https://www.kaggle.com/competitions/deep-learning-mini-project-spring-24-nyu 

The best result (84.7% on Kaggle test) we have is using ResNet5M model with these hyperparameters: 

    total_train_size = 60000 (no validation set)
    train_batch_size = 400
    epochs = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 
                            lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
