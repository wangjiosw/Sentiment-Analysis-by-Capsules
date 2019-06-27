import torch.optim as optim
import numpy as np
import os
import torch


def train(train_iter, val_iter, test_iter, model, runModel, n_epoch=30, lr=0.00001, print_val_every_num=20):
    if os.path.exists('params.pkl'):
        model.load_state_dict(torch.load('params.pkl'))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0

    for epoch in range(n_epoch):

        for batch_idx, batch in enumerate(train_iter):
            train_x = batch.INPUT
            target = batch.OUTPUT
            optimizer.zero_grad()

            loss, y_pre, acc = runModel(model, train_x, target)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % print_val_every_num == 0:
                print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f'
                      % (epoch, batch_idx, loss, acc))

        val_accs = []
        for batch_idx, batch in enumerate(val_iter):
            val_x = batch.INPUT
            target = batch.OUTPUT

            loss, y_pre, acc = runModel(model, val_x, target)
            val_accs.append(acc)

        acc = np.array(val_accs).mean()
        if acc > best_val_acc:
            print('val acc : %.4f > %.4f saving model' % (acc, best_val_acc))
            torch.save(model.state_dict(), 'params.pkl')
            best_val_acc = acc
        print('val acc: %.4f'%acc)

    test_accs = []
    for batch_idx, batch in enumerate(test_iter):
        test_x = batch.INPUT
        target = batch.OUTPUT

        loss, y_pre, acc = runModel(model, test_x, target)
        acc = torch.mean((torch.tensor(y_pre == batch.OUTPUT, dtype=torch.float)))
        test_accs.append(acc)

    acc = np.array(test_accs).mean()
    print('test acc: %.4f'%acc)