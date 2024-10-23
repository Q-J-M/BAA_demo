from utils import to_var

def train(image, label, gender, model, optimizer, criterion, epoch):

    train_data,train_ref, train_gender= image, label, gender

    model.train()

    train_image = to_var(train_data).detach()
    train_gender = to_var(train_gender.view(-1, 1)).detach()
    train_age = to_var(train_ref.view(-1, 1)).detach()

    optimizer.zero_grad()

    out = model(train_image, train_gender)

    loss = criterion(out, train_age)

    loss.backward()
    optimizer.step()

    loss_value = loss.detach().cpu().item()

    with open('train_loss.txt', 'a') as f:
        f.write(str(epoch) + ',' + str(loss_value) + '\n')

    # print('Epoch [%d/%d], Loss: %.4f'
    #       % (epoch,
    #          n_epochs,
    #          loss,
    #          )
    #       )